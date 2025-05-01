import asyncio
from typing import Optional

from openai import AsyncOpenAI, OpenAIError  # Use AsyncOpenAI for async operations

# Directly import the config instance
from config import config
from logger_config import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Handles interaction with an OpenAI-compatible LLM API."""

    def __init__(self):
        self.enabled = config.get("llm.enabled", False)
        self.api_key = config.get("llm.api_key")
        self.base_url = config.get("llm.base_url")  # Can be None
        self.model = config.get("llm.model", "gpt-3.5-turbo")
        # This now gets the prompt loaded from file by config.py, or the fallback/default if loading failed.
        self.system_prompt = config.get(
            "llm.system_prompt", "You are a helpful assistant."
        )  # Fallback default
        self.temperature = config.get("llm.temperature", 0.7)
        self.max_tokens = config.get("llm.max_tokens", 150)
        self.few_shot_examples = config.get("llm.few_shot_examples", [])

        self.client: Optional[AsyncOpenAI] = None

        # Validate few_shot_examples structure
        if self.enabled and self.few_shot_examples:
            if not isinstance(self.few_shot_examples, list):
                logger.warning(
                    "LLM 'few_shot_examples' is not a list in config. Disabling examples."
                )
                self.few_shot_examples = []
            else:
                valid_examples = []
                for i, example in enumerate(self.few_shot_examples):
                    if (
                        isinstance(example, dict)
                        and "user" in example
                        and "assistant" in example
                    ):
                        valid_examples.append(example)
                    else:
                        logger.warning(
                            f"Invalid structure for few_shot_example at index {i}. Skipping. Expected dict with 'user' and 'assistant' keys."
                        )
                self.few_shot_examples = valid_examples  # Keep only valid ones
                if self.few_shot_examples:
                    logger.info(
                        f"Loaded {len(self.few_shot_examples)} valid LLM few-shot examples."
                    )

        if self.enabled:
            if not self.api_key:
                logger.error(
                    "LLMClient: LLM is enabled but API key is missing. LLM processing will be disabled."
                )
                self.enabled = False  # Disable if key is missing
            else:
                try:
                    logger.info(
                        f"Initializing LLMClient for model '{self.model}' (Base URL: {self.base_url or 'Default'})."
                    )
                    self.client = AsyncOpenAI(
                        api_key=self.api_key,
                        base_url=self.base_url,  # Pass None if not specified, client handles default
                    )
                except Exception as e:
                    logger.error(
                        f"LLMClient: Failed to initialize AsyncOpenAI client: {e}",
                        exc_info=True,
                    )
                    self.enabled = False  # Disable on initialization error

    async def process_text(self, text: str) -> Optional[str]:
        """
        Sends text to the configured LLM for processing.

        Args:
            text: The input text (e.g., from STT).

        Returns:
            The processed text from the LLM, or None if processing failed,
            is disabled, or the client is not initialized.
        """
        if not self.enabled or not self.client:
            logger.debug(
                "LLMClient.process_text called but LLM is disabled or client not initialized."
            )
            return None

        logger.debug(f"LLMClient: Processing text: '{text[:50]}...'")
        try:
            # Construct the messages list dynamically
            messages = [{"role": "system", "content": self.system_prompt}]

            # Add few-shot examples if available
            if self.few_shot_examples:
                logger.debug(
                    f"Adding {len(self.few_shot_examples)} few-shot examples to LLM request."
                )
                for example in self.few_shot_examples:
                    # Ensure correct roles are used as per OpenAI API
                    messages.append({"role": "user", "content": example["user"]})
                    messages.append(
                        {"role": "assistant", "content": example["assistant"]}
                    )

            # Add the actual user input after system prompt and examples
            messages.append({"role": "user", "content": text})

            logger.debug(
                f"LLM API call messages: {messages}"
            )  # Log the full message structure for debugging

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # Pass the constructed messages list
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Extract the response content
            if response.choices and response.choices[0].message:
                processed_text = response.choices[0].message.content
                if processed_text:
                    processed_text = processed_text.strip()
                    logger.info(
                        f"LLMClient: Received processed text: '{processed_text[:50]}...'"
                    )
                    return processed_text
                else:
                    logger.warning(
                        "LLMClient: Received empty response content from LLM."
                    )
                    return None  # Treat empty response as failure
            else:
                logger.warning(
                    "LLMClient: LLM response did not contain expected choices or message."
                )
                return None

        except OpenAIError as e:
            logger.error(
                f"LLMClient: OpenAI API error during processing: {e}", exc_info=True
            )
            return None
        except Exception as e:
            logger.error(
                f"LLMClient: Unexpected error during LLM processing: {e}", exc_info=True
            )
            return None
