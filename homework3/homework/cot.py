from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        messages = [
            {"role": "system", "content": "Convert units. Show work. Always end with <answer>NUMBER</answer>."},
            {"role": "user", "content": "How many grams in 2 kg?"},
            {"role": "assistant", "content": "1 kg = 1000 g. 2 * 1000 = 2000. <answer>2000.0</answer>"},
            {"role": "user", "content": "How many seconds in 3 hours?"},
            {"role": "assistant", "content": "1 hour = 3600 seconds. 3 * 3600 = 10800. <answer>10800.0</answer>"},
            {"role": "user", "content": "How many feet in 5 yards?"},
            {"role": "assistant", "content": "1 yard = 3 feet. 5 * 3 = 15. <answer>15.0</answer>"},
            {"role": "user", "content": question},
        ]

        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        

def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
