from unsloth import FastLanguageModel
import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

class Model():
    def __init__(self, model_path, use_unsloth=False):
        """
        Initialize the model with the appropriate type (text or vision).
        
        Args:
            model_path (str): Path to the fine-tuned model directory.
            use_unsloth (bool): Whether to use Unsloth's FastLanguageModel.
        """
        self.use_unsloth = use_unsloth
        self.is_vision_model = "vision" in model_path.lower()  # Determine if it's a vision-based model
        self.name = model_path
        print(f"Model is vision-based: {self.is_vision_model}")

        if not self.is_vision_model:
            try:
                if self.use_unsloth:
                    print("Loading Unsloth FastLanguageModel...")
                    # Load Unsloth model
                    self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                        model_name=model_path,
                        max_seq_length=512,  # Adjust as needed
                        dtype=torch.float16,  # Use FP16 for faster inference
                        load_in_4bit=True,  # Enable 4-bit quantization
                    )
                    FastLanguageModel.for_inference(self.model)  # Enable 2x faster inference
                else:
                    print("Loading Hugging Face text model...")
                    # Load Hugging Face model
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                    self.model = AutoModelForCausalLM.from_pretrained(model_path)

                    # Set the pad_token_id to the eos_token_id if not already set
                    if self.tokenizer.pad_token_id is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token

                # Move model to the appropriate device
                #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                #self.model = self.model.to(self.device)
                print("Text model loaded successfully.")
            except Exception as e:
                print(f"Error loading text model or tokenizer from {model_path}: {e}")
                raise
        else:
            try:
                # Vision-based model initialization
                print("Loading vision model...")
                self.processor = AutoProcessor.from_pretrained(model_path)
                # Define quantization settings
                self.bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,  # Use 4-bit quantization (Set to True for lower VRAM usage)
                    bnb_4bit_compute_dtype=torch.float16,  # Set compute dtype to FP16
                    bnb_4bit_use_double_quant=True,  # Enable double quantization for better memory efficiency
                )
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_path,
                    quantization_config=self.bnb_config,
                    torch_dtype=torch.float16 if "fp16" in model_path else torch.float32,
                    low_cpu_mem_usage=True # Enable low CPU memory usage
                )
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Using device: {self.device}")
                self.model = self.model.to(self.device)
                self.model = torch.compile(self.model) 
            except Exception as e:
                print(f"Error loading vision model or processor from {model_path}: {e}")
                raise

    async def ask(self, input_data, max_length=50):
        """
        Ask the model a question or provide an image and return the response.
        
        Args:
            input_data (str or PIL.Image): The input text or image to process.
            max_length (int): The maximum length of the response (for text models).
        
        Returns:
            str: The response from the model.
        """
        # Run the model inference in a separate thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        if self.is_vision_model:
            if isinstance(input_data, str):
                response = await loop.run_in_executor(None, self._vision_model_generate_response_text, input_data)
            else:
                #response = await loop.run_in_executor(None, self._generate_response_text_vision, input_data)
                response = "Need to implement..."
        else:
            response = await loop.run_in_executor(None, self._generate_response_text, input_data, max_length)
        return response

    def _generate_response_text(self, question, max_length=50):
        """
        Generate a response from a text-based model.
        
        Args:
            question (str): The input text to generate a response for.
            max_length (int): The maximum length of the response.
        
        Returns:
            str: The generated response from the model.
        """
        try:
            if self.use_unsloth:
                # Define Alpaca-style prompt
                alpaca_prompt = "### Instruction:\n{}\n\n### Response:\n{}"
                prompt = alpaca_prompt.format(question, "")

                # Tokenize input
                inputs = self.tokenizer([prompt], return_tensors="pt").to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

                # Generate response
                text_streamer = TextStreamer(self.tokenizer)
                output = self.model.generate(
                    **inputs,
                    #streamer=text_streamer,
                    max_new_tokens=96,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode the response
                response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                response = response.split("### Response:")[1].strip()
                print(f"Question: {question}")
                print(f"Response: {response}")
            else:
                # Tokenize input
                inputs = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True)
                input_ids = inputs["input_ids"].to(self.model.device)
                attention_mask = inputs["attention_mask"].to("cuda")

                # Generate response
                output = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.5,
                )

                # Decode the response
                response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                print(f"Question: {question}")
                print(f"Response: {response}")
            
            # Return the response
            return response

        except Exception as e:
            print(f"Error generating text response: {e}")
            return "Error generating response. Please try again."

    def _vision_model_generate_response_text(self, text):
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_length=200
            )  # Adjust length as needed
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

    def _generate_response_vision(self, image):
        """
        Generate a response from a vision-based model.
        
        Args:
            image (PIL.Image): The input image to process.
        
        Returns:
            str: The generated response from the model.
        """
        try:
            # Preprocess the image
            inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
            
            # Generate response
            output = self.model.generate(**inputs)
            
            # Decode the response
            response = self.processor.tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"Generated response for image: {response}")
            return response
        
        except Exception as e:
            print(f"Error generating vision response: {e}")
            return "Error generating response. Please try again."

