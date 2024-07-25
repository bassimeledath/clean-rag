import weave
from openai import OpenAI
import json
import asyncio


class RAGEvaluator:
    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
        self.openai_client = OpenAI()

    @weave.op()
    async def context_precision_score(self, question, model_output):
        context_precision_prompt = """Given question, answer and context verify if the context was useful in arriving at the given answer. Give verdict as "1" if useful and "0" if not with json output. Output in only valid JSON format.
        question: {question}
        context: {context}
        answer: {answer}
        verdict: """

        prompt = context_precision_prompt.format(
            question=question,
            context=model_output['context'],
            answer=model_output['answer'],
        )
        response = self.openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_object"
            }
        )
        response_message = response.choices[0].message
        response = json.loads(response_message.content)
        return {
            "verdict": int(response["verdict"]) == 1,
        }

    @weave.op()
    async def evaluate_question(self, question):
        result = self.rag_engine.predict(question)
        score = await self.context_precision_score(question, result)
        return {**result, "score": score["verdict"]}

    @weave.op()
    async def evaluate_questions(self, questions):
        results = []
        for question in questions:
            result = await self.evaluate_question(question)
            results.append(result)
            print(f"Question: {question}")
            print(f"Answer: {result['answer']}")
            print(f"Context: {result['context']}")
            print(f"Score: {'Useful' if result['score'] else 'Not useful'}")
            print("--------------------")

        overall_score = sum(result['score']
                            for result in results) / len(results)
        print(f"Overall score: {overall_score:.2f}")
        return results, overall_score
