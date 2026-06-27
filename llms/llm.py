from PIL import Image
from pathlib import Path
import base64
from io import BytesIO


def _encode_image(image_path):
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

        from openai import OpenAI

        if self.model_name.startswith("gpt"):
            self.model = OpenAI()
        else:
            if base_url is None:
                raise ValueError("base_url is required when using a non-gpt model through vLLM.")
            if api_key is None:
                raise ValueError("api_key is required when using a non-gpt model through vLLM.")
            self.model = OpenAI(base_url=base_url, api_key=api_key)
            
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
        return completion.choices[0].message.content

if __name__ == '__main__':
    llm = LLM('gpt-4o')
    response = llm.generate(query='describe in 3 words',image=['image_path'])
    print(response)
