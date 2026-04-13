import os
import base64
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# The multimodal model available on Groq
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def get_image_description(image_bytes: bytes) -> str:
    """
    Sends image bytes to Groq Vision model and returns a descriptive text.
    """
    if not image_bytes:
        return ""

    # Encode image to base64
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    try:
        completion = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Summarize the contents of this image in detail. "
                                "Describe what is happening, any text visible, objects, people, technical diagrams, or screenshots. "
                                "Make the description keyword-rich for a search engine."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=512,
            temperature=0.1,  # Low temperature for more factual descriptions
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling Groq Vision API: {e}")
        return f"[Image description failed: {str(e)}]"

if __name__ == "__main__":
    # Test script part
    print("Testing vision module...")
    # Add a real test here if needed, but the logic is verified by scratch/test_vision.py
    pass
