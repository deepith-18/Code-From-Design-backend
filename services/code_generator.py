import os
import json
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# ==========================================
# LOAD ENV
# ==========================================
load_dotenv()
logger = logging.getLogger(__name__)

# ==========================================
# CODE GENERATOR
# ==========================================

class CodeGenerator:
    def __init__(self):
        # Gemini API Key
        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # FIXED: Explicitly pass the model name here
        # 'gemini-3-flash-preview' is the current best free-tier vision model
        self.model = genai.GenerativeModel(
            model_name='gemini-3-flash-preview'
        )

        logger.info("Gemini initialized successfully with gemini-3-flash-preview")

    # ==========================================
    # GENERATE CODE
    # ==========================================

    def generate(self, image_path, elements=None):
        try:
            logger.info("Generating code using Gemini...")

            # ==========================================
            # PROMPT
            # ==========================================
            prompt = """
You are an expert frontend engineer.
Analyze this UI screenshot carefully.

Generate:
1. HTML
2. CSS
3. React component

Requirements:
- Match layout accurately
- Preserve colors, spacing, and typography
- Use semantic HTML and plain CSS (No Tailwind)
- No explanations or markdown formatting outside the JSON
- Responsive design

Return ONLY a valid JSON object:
{
  "html": "...",
  "css": "...",
  "react": "..."
}
"""

            # ==========================================
            # READ IMAGE
            # ==========================================
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()

            # ==========================================
            # GEMINI REQUEST
            # ==========================================
            # Added generation_config to ensure valid JSON output
            response = self.model.generate_content(
                contents=[
                    prompt,
                    {
                        "mime_type": "image/png",
                        "data": image_bytes
                    }
                ],
                generation_config={"response_mime_type": "application/json"}
            )

            # With response_mime_type, response.text should be clean JSON
            raw_output = response.text.strip()

            logger.info(f"Gemini Output Received (showing first 100 chars): {raw_output[:100]}")

            # ==========================================
            # PARSE JSON
            # ==========================================
            try:
                # If the model still includes markdown backticks, this cleans them
                if raw_output.startswith("```"):
                    raw_output = raw_output.replace("```json", "").replace("```", "").strip()
                
                parsed = json.loads(raw_output)

                # Ensure all keys exist
                parsed.setdefault("html", "")
                parsed.setdefault("css", "")
                parsed.setdefault("react", "")

                logger.info("Code generation successful")
                return parsed

            except json.JSONDecodeError as json_error:
                logger.error(f"JSON Parse Error: {json_error}")
                return {
                    "html": "",
                    "css": "",
                    "react": raw_output,
                    "error": "Model returned invalid JSON format"
                }

        except Exception as e:
            logger.error(f"Gemini generation failed: {str(e)}")
            return {
                "html": "",
                "css": "",
                "react": "",
                "error": str(e)
            }