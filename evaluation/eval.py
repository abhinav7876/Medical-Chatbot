from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json
import os
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
chatModel = ChatOpenAI(model="gpt-4o")

def evaluate_with_threshold(query, context, response):
    EVALUATION_PROMPT = """
        You are an expert evaluator that scores AI-generated answers on three key metrics.

        Given:
        - Question: {question}
        - Retrieved Context: {context}
        - Answer: {answer}

        Evaluate according to:
        1. Faithfulness — accuracy based on context.
        2. Relevance — how well the answer matches the question.
        3. Conciseness — clarity and brevity.

        Return only valid JSON:
        {{
        "faithfulness": <float 0-1>,
        "relevance": <float 0-1>,
        "conciseness": <float 0-1>
        }}
        """
    eval_prompt = EVALUATION_PROMPT.format(
    question=query,
    context=context,
    answer=response
)
    
    eval_result = chatModel.invoke(eval_prompt).content
    try:
        cleaned = eval_result.replace("```json", "").replace("```", "").strip()
        scores = json.loads(cleaned)
        print("scores are: ",scores)
    except json.JSONDecodeError:
        print("entered into json exception")
        scores = {"faithfulness": 0.0, "relevance": 0.0, "conciseness": 0.0}
    thresholds = {"faithfulness": 0.8, "relevance": 0.75, "conciseness": 0.7}
    low_scores = {m: v for m, v in scores.items() if v < thresholds[m]}

    if low_scores:
        print(f"Low metrics detected: {low_scores}")
        refined_query = (
            f"{query}\nPlease refine your answer using only verified, factual information. "
            f"Ensure faithfulness and relevance. Be concise."
        )
        refined_response = chatModel.invoke(refined_query).content
        return refined_response, scores
    else:
        return response,scores