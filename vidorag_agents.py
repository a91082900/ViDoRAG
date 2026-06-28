from PIL import Image
import os
import json
import sys

from agent.agent_prompt import seeker_prompt,inspector_prompt,answer_prompt
from agent.map_dict import arrangement_map_dict,page_map_dict_normal,page_map_dict

from utils.parse_tool import extract_json
from utils.image_preprosser import concat_images_with_bbox

MAX_AGENT_ATTEMPTS = 3


class InvalidImageIds(ValueError):
    def __init__(self, image_ids, image_count, allow_empty=True):
        self.image_ids = image_ids
        self.image_count = image_count
        self.allow_empty = allow_empty
        super().__init__(
            f"invalid image ids {image_ids!r}; expected "
            f"{'zero or more' if allow_empty else 'one or more'} unique integers "
            f"between 0 and {image_count - 1}"
        )


def _image_names(images):
    return [os.path.basename(image) for image in images]


def _validate_image_ids(image_ids, image_count, allow_empty=True):
    if not isinstance(image_ids, list):
        raise InvalidImageIds(image_ids, image_count, allow_empty)

    validated_ids = []
    seen_ids = set()
    for image_id in image_ids:
        if (
            isinstance(image_id, bool)
            or not isinstance(image_id, int)
            or image_id < 0
            or image_id >= image_count
            or image_id in seen_ids
        ):
            raise InvalidImageIds(image_ids, image_count, allow_empty)
        seen_ids.add(image_id)
        validated_ids.append(image_id)

    if not allow_empty and not validated_ids:
        raise InvalidImageIds(image_ids, image_count, allow_empty)
    return validated_ids


def _image_id_retry_instruction(image_count):
    return (
        "\n\nYour previous response contained invalid image IDs. "
        f"Use only unique integer IDs from 0 to {image_count - 1}. "
        "Return the complete response as JSON."
    )


class Seeker:
    def __init__(self, vlm):
        self.vlm = vlm
        self.seeker_multi_image = False

        if self.seeker_multi_image:
            self.page_map = page_map_dict_normal
        else:
            self.page_map = page_map_dict
    def run(self, query=None, images_path=None, feedback=None):

        if query is not None and images_path is not None:
            self.buffer_images = images_path
            self.query = query
            prompt = seeker_prompt.replace('{question}', self.query).replace('{page_map}', self.page_map[len(self.buffer_images)])
        
        elif feedback is not None:
            additional_information = self.query + '\n\n## Additional Information\n' + feedback
            prompt = seeker_prompt.replace('{question}', additional_information).replace('{page_map}', self.page_map[len(self.buffer_images)])        

        if self.seeker_multi_image:
            input_images = self.buffer_images
        else:
            input_images = [concat_images_with_bbox(self.buffer_images, arrangement=arrangement_map_dict[len(self.buffer_images)], scale=1, line_width=40)]
        source_images = list(self.buffer_images)

        times = 0
        fallback_used = False
        invalid_image_ids = None
        retry_for_invalid_ids = False
        while times < MAX_AGENT_ATTEMPTS:
            times += 1
            if retry_for_invalid_ids:
                select_response = self.vlm.generate(
                    query=prompt + _image_id_retry_instruction(len(self.buffer_images)),
                    image=input_images)
            else:
                select_response = self.vlm.generate(query=prompt, image=input_images)
            retry_for_invalid_ids = False
            print(select_response)
            try:
                select_response_json = extract_json(select_response)
                reason = select_response_json.get('reason', None)
                summary = select_response_json.get('summary', None)
                select_page_num = select_response_json.get('choice', None)
                if reason is None or summary is None:
                    raise Exception(f'select json format error: length: {len(self.buffer_images)}')
                try:
                    select_page_num = _validate_image_ids(
                        select_page_num,
                        len(self.buffer_images),
                    )
                except InvalidImageIds as e:
                    invalid_image_ids = e.image_ids
                    if times < MAX_AGENT_ATTEMPTS:
                        retry_for_invalid_ids = True
                        raise
                    select_page_num = list(range(len(self.buffer_images)))
                    fallback_used = True
                    print(
                        "seeker image ID fallback: "
                        f"using all {len(self.buffer_images)} images"
                    )

                selected_images = [self.buffer_images[page] for page in select_page_num]
                self.buffer_images = [image for image in self.buffer_images if image not in selected_images]

            except Exception as e:
                print('seeker error')
                print(e)
                print(select_response)
                continue
            break
        else:
            raise Exception('seeker time out')
        print("\n\nSeeker:")
        print(f"Selected images: {selected_images}")
        print(f"Summary: {summary}")
        print(f"Reason: {reason}\n")
        self.last_action = {
            "action": "select_evidence",
            "input_images": _image_names(source_images),
            "selected_images": _image_names(selected_images),
            "summary": summary,
            "reason": reason,
            "attempts": times,
            "fallback_used": fallback_used,
        }
        if fallback_used:
            self.last_action["invalid_image_ids"] = invalid_image_ids
        return selected_images, summary, reason

class Inspector:
    def __init__(self, vlm):
        self.vlm = vlm
        self.inspector_multi_image = False

        if self.inspector_multi_image:
            self.page_map = page_map_dict_normal
        else:
            self.page_map = page_map_dict

        self.buffer_images = []

    def run(self, query, images_path):
        # answer or not, (images, candidate)/feedback
        if len(self.buffer_images) == 0 and len(images_path) == 0:
            self.last_action = {
                "action": "no_evidence",
                "input_images": [],
            }
            return None, None, None
        elif len(images_path) == 0:
            self.last_action = {
                "action": "send_to_synthesizer",
                "input_images": _image_names(self.buffer_images),
                "referenced_images": _image_names(self.buffer_images),
                "candidate_answer": None,
            }
            return 'synthesizer', None, self.buffer_images
        elif len(images_path) != 0:
            self.buffer_images.extend(images_path)

        if self.inspector_multi_image:
            input_images = self.buffer_images
        else:
            input_images = [concat_images_with_bbox(self.buffer_images, arrangement=arrangement_map_dict[len(self.buffer_images)], scale=1, line_width=40)]

        prompt = inspector_prompt.replace('{question}',query).replace('{page_map}',self.page_map[len(self.buffer_images)])

        times = 0
        retry_for_invalid_ids = False
        while times < MAX_AGENT_ATTEMPTS:
            times +=1

            request_prompt = prompt
            if retry_for_invalid_ids:
                request_prompt += _image_id_retry_instruction(len(self.buffer_images))
            response = self.vlm.generate(query=request_prompt,image=input_images)
            retry_for_invalid_ids = False
            print(response)
            try:
                response_json = extract_json(response)
                # thought
                reason = response_json.get('reason',None)
                # if feedback
                info = response_json.get('information',None)
                choice = response_json.get('choice',None)
                # can answer
                answer = response_json.get('answer',None)
                ref = response_json.get('reference',None)

                print("\n\nInspector:")
                print(f"Reason: {reason}")
                print(f"Information: {info}")
                print(f"Choice: {choice}")
                print(f"Answer: {answer}")
                print(f"Reference: {ref}\n")

                if reason is None:
                    raise Exception('answer no reason')
                elif answer is not None and ref is not None:
                    fallback_used = False
                    invalid_image_ids = None
                    try:
                        ref = _validate_image_ids(
                            ref,
                            len(self.buffer_images),
                            allow_empty=False,
                        )
                    except InvalidImageIds as e:
                        invalid_image_ids = e.image_ids
                        if times < MAX_AGENT_ATTEMPTS:
                            retry_for_invalid_ids = True
                            raise
                        ref = list(range(len(self.buffer_images)))
                        fallback_used = True
                        print(
                            "inspector reference fallback: "
                            f"using all {len(self.buffer_images)} images"
                        )
                    if len(ref) == len(self.buffer_images):
                        self.last_action = {
                            "action": "answer",
                            "input_images": _image_names(self.buffer_images),
                            "referenced_images": _image_names(self.buffer_images),
                            "reason": reason,
                            "answer": answer,
                            "attempts": times,
                            "fallback_used": fallback_used,
                        }
                        if fallback_used:
                            self.last_action["invalid_image_ids"] = invalid_image_ids
                        return 'answer', answer, self.buffer_images
                    else:
                        ref_images = [self.buffer_images[page] for page in ref]
                        self.last_action = {
                            "action": "send_to_synthesizer",
                            "input_images": _image_names(self.buffer_images),
                            "referenced_images": _image_names(ref_images),
                            "reason": reason,
                            "candidate_answer": answer,
                            "attempts": times,
                            "fallback_used": fallback_used,
                        }
                        if fallback_used:
                            self.last_action["invalid_image_ids"] = invalid_image_ids
                        return 'synthesizer', answer, ref_images
                elif info is not None and choice is not None:
                    fallback_used = False
                    invalid_image_ids = None
                    try:
                        choice = _validate_image_ids(
                            choice,
                            len(self.buffer_images),
                        )
                    except InvalidImageIds as e:
                        invalid_image_ids = e.image_ids
                        if times < MAX_AGENT_ATTEMPTS:
                            retry_for_invalid_ids = True
                            raise
                        choice = list(range(len(self.buffer_images)))
                        fallback_used = True
                        print(
                            "inspector choice fallback: "
                            f"retaining all {len(self.buffer_images)} images"
                        )
                    inspected_images = list(self.buffer_images)
                    self.buffer_images = [self.buffer_images[page] for page in choice]
                    self.last_action = {
                        "action": "request_more_evidence",
                        "input_images": _image_names(inspected_images),
                        "retained_images": _image_names(self.buffer_images),
                        "reason": reason,
                        "information_needed": info,
                        "attempts": times,
                        "fallback_used": fallback_used,
                    }
                    if fallback_used:
                        self.last_action["invalid_image_ids"] = invalid_image_ids
                    return 'seeker', info, self.buffer_images
                else:
                    raise Exception('inspector response format error')
            
            except Exception as e:
                print(e)
                print("inspector")
                continue
        raise Exception('Inspector time out')

class Synthesizer:
    def __init__(self, vlm):
        self.vlm = vlm
        self.synthesizer_multi_image = False
        if self.synthesizer_multi_image:
            self.page_map = page_map_dict_normal
        else:
            self.page_map = page_map_dict

    def run(self, query, candidate_answer, ref_images):
        if candidate_answer is not None:
            query = query + '\n\n## Related Information\n' + candidate_answer
        prompt = answer_prompt.replace('{question}',query).replace('{page_map}',self.page_map[len(ref_images)])

        if self.synthesizer_multi_image:
            input_images = ref_images
        else:
            input_images = [concat_images_with_bbox(ref_images, arrangement=arrangement_map_dict[len(ref_images)], scale=1, line_width=40)]
        
        while True:
            final_answer_response = self.vlm.generate(query=prompt,image=input_images)
            print("\n\nSynthesizer:")
            print(f"Final Answer Response: ", end="")
            print(final_answer_response)
            try:
                final_answer_response_json = extract_json(final_answer_response)
                reason = final_answer_response_json.get('reason',None)
                answer = final_answer_response_json.get('answer',None)
                if reason is None or answer is None :
                    raise Exception('Synthesizer time out')
                self.last_action = {
                    "action": "synthesize_answer",
                    "referenced_images": _image_names(ref_images),
                    "candidate_answer": candidate_answer,
                    "reason": reason,
                    "answer": answer,
                }
                return reason, answer
            except Exception as e:
                print(e)
                print(final_answer_response)
                print("answer")
                continue

class ViDoRAG_Agents:
    def __init__(self, vlm):
        self.seeker = Seeker(vlm)
        self.inspector = Inspector(vlm)
        self.synthesizer = Synthesizer(vlm)

    def run_agent(self, query, images_path, return_trace=False):
        # initial
        self.seeker.buffer_images = None
        self.inspector.buffer_images = []
        trace = []

        def record(agent, action):
            trace.append({
                "step": len(trace) + 1,
                "agent": agent,
                **dict(action),
            })

        selected_images, summary, reason = self.seeker.run(query=query, images_path=images_path)
        record("seeker", self.seeker.last_action)
        # iter
        while True:
            status, information, images = self.inspector.run(query, selected_images)
            record("inspector", self.inspector.last_action)
            if status == 'answer':
                if return_trace:
                    return information, trace
                return information
            elif status == 'synthesizer':
                reason, answer = self.synthesizer.run(query, information, images)
                record("synthesizer", self.synthesizer.last_action)
                if return_trace:
                    return answer, trace
                return answer
            elif status == 'seeker':
                if not self.seeker.buffer_images:
                    reason, answer = self.synthesizer.run(query, None, images)
                    record("synthesizer", self.synthesizer.last_action)
                    if return_trace:
                        return answer, trace
                    return answer
                selected_images, summary, reason = self.seeker.run(feedback=information)
                record("seeker", self.seeker.last_action)
                continue
            else:
                print('No related information')
                if return_trace:
                    return None, trace
                return None

if __name__ == '__main__':
    from llms.llm import LLM
    vlm = LLM('qwen-vl-max')
    agent = ViDoRAG_Agents(vlm)
    re=agent.run_agent(query='Who is Tim?', images_path=['./data/ExampleDataset/img/00a76e3a9a36255616e2dc14a6eb5dde598b321f_1.jpg','./data/ExampleDataset/img/00a76e3a9a36255616e2dc14a6eb5dde598b321f_2.jpg','./data/ExampleDataset/img/00a76e3a9a36255616e2dc14a6eb5dde598b321f_3.jpg','./data/ExampleDataset/img/00a76e3a9a36255616e2dc14a6eb5dde598b321f_4.jpg'])
    print(re)
