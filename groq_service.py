import logging
from typing import Any, Callable


MODEL_CASCADE = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]


class GroqGenerationError(Exception):
    """Raised when every configured Groq model fails."""


async def generate_with_model_cascade(
    client: Any,
    messages: list[dict[str, str]],
    logger: logging.Logger,
    validator: Callable[[Any, str], Any] | None = None,
    **completion_options: Any,
) -> tuple[Any, str]:
    last_error = None

    for model_name in MODEL_CASCADE:
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                **completion_options,
            )
            result = validator(response, model_name) if validator else response
            return result, model_name
        except Exception as error:
            last_error = error
            logger.exception(
                "Model '%s' failed: %s. Cascading to next model...",
                model_name,
                error,
            )

    raise GroqGenerationError(
        "All models in cascade failed. "
        f"Last error: {last_error}"
    )
