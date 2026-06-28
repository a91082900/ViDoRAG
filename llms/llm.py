from PIL import Image
from pathlib import Path
import base64
from io import BytesIO
import threading


def auto_resize_image(image, max_pixels=1280*28*28):
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        width, height = image.size
        total_pixels = width * height
        if total_pixels > max_pixels:
            scale = (max_pixels / total_pixels) ** 0.5
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.LANCZOS)
    else:
        raise ValueError("Input must be a PIL Image or a valid image file path.")
    return image

def _encode_image(image_path):
    image_path = auto_resize_image(image_path)
    if isinstance(image_path, Image.Image):
        buffered = BytesIO()
        image_path.save(buffered, format="JPEG")
        img_data = buffered.getvalue()
        base64_encoded = base64.b64encode(img_data).decode("utf-8")
        return base64_encoded
    else:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


class LLM:
    def __init__(self, model_name, base_url=None, api_key=None, max_tokens=1028):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._usage_state = threading.local()

        from openai import OpenAI

        if self.model_name.startswith("gpt"):
            self.model = OpenAI()
        else:
            if base_url is None:
                raise ValueError("base_url is required when using a non-gpt model through vLLM.")
            if api_key is None:
                raise ValueError("api_key is required when using a non-gpt model through vLLM.")
            self.model = OpenAI(base_url=base_url, api_key=api_key)

    def reset_usage(self):
        self._usage_state.value = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
        }

    def get_usage(self):
        usage = getattr(self._usage_state, "value", None)
        if usage is None:
            self.reset_usage()
            usage = self._usage_state.value
        return dict(usage)

    def _record_usage(self, completion):
        usage = self.get_usage()
        response_usage = getattr(completion, "usage", None)
        usage["prompt_tokens"] += int(getattr(response_usage, "prompt_tokens", 0) or 0)
        usage["completion_tokens"] += int(getattr(response_usage, "completion_tokens", 0) or 0)
        usage["total_tokens"] += int(getattr(response_usage, "total_tokens", 0) or 0)
        usage["requests"] += 1
        self._usage_state.value = usage

    def generate(self, **kwargs):
        query = kwargs.get("query", "")
        image = kwargs.get("image", "")

        content = [{
            "type": "text",
            "text": query
        }]
        if image != "":
            for img in image:
                if isinstance(img, Image.Image):
                    base64_image = _encode_image(img)
                else:
                    filepath = Path(img).resolve().as_posix()
                    base64_image = _encode_image(filepath)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
        completion = self.model.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=self.max_tokens,
        )
        self._record_usage(completion)
        return completion.choices[0].message.content

if __name__ == '__main__':
    llm = LLM('gpt-4o')
    response = llm.generate(query='describe in 3 words',image=['image_path'])
    print(response)
