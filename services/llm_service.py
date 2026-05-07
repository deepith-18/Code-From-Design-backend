import logging
import base64


logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        # Use your new Fine-grained token here
        self.token = "" 
        self.client = InferenceClient(api_key=self.token)
        
        # CHANGED: Using the 2.5 version which is explicitly allowed by your HF provider
        self.model = "Qwen/Qwen2.5-VL-7B-Instruct"

    def generate_code_from_image(self, image_path):
        try:
            logger.info(f"Using {self.model} (Version 2.5) via Hugging Face...")
            
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")

            prompt = """
           You are a senior frontend engineer specialized in React (Vite) and Tailwind CSS.

Your task is to analyze the provided UI image and generate a production-ready React functional component.

STRICT RULES:
1. Output ONLY valid JSON.
2. The JSON must contain ONLY:
   {
     "react": "complete React component code"
   }
3. No explanations.
4. No markdown.
5. No extra keys.
6. Match the layout hierarchy exactly as seen in the image.
7. Preserve spacing proportions using Tailwind scale (p-4, p-6, gap-4, gap-6 etc).
8. Use proper flex or grid layout depending on structure.
9. Do not omit any visible element.

Generate a complete export default component.

            }
            """

            response = self.client.chat_completion(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url", 
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                max_tokens=3000
            )
            
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"HF Error: {str(e)}")
            return f"ERROR: {str(e)}"