
import time

from retrievers import (
    similarity_search,
    mmr_search,
    multi_query_search,
    bm25_search,
    hybrid_search
)


# Evaluation Dataset
test_cases = [
    {
        "question": "What is backpropagation?",
        # Add manually verified relevant page numbers later
        "expected_pages": []
    }
]


# All Retrieval Algorithms
retrievers = {
    "Similarity": similarity_search,
    "MMR": mmr_search,
    "Multi-Query": multi_query_search,
    "BM25": bm25_search,
    "Hybrid": hybrid_search
}


# Extract Page Numbers
def get_pages(documents):
    return [
        str(doc.metadata.get("page_label"))
        for doc in documents[:4]
    ]


# Calculate Evaluation Metrics
def calculate_metrics(retrieved_pages, expected_pages, k=4):

    if not expected_pages:
        return None, None, None

    expected_pages = set(map(str, expected_pages))

    relevant = [
        page for page in retrieved_pages
        if page in expected_pages
    ]

    precision = len(relevant) / k
    recall = len(set(relevant)) / len(expected_pages)

    reciprocal_rank = 0

    for rank, page in enumerate(retrieved_pages, 1):
        if page in expected_pages:
            reciprocal_rank = 1 / rank
            break

    return precision, recall, reciprocal_rank


# Main Evaluation Function
def evaluate_retrievers(test_cases, k=4):

    for test in test_cases:

        question = test["question"]
        expected_pages = test["expected_pages"]

        print("\n" + "=" * 80)
        print("QUESTION:", question)
        print("=" * 80)

        results_table = []

        for name, retriever in retrievers.items():

            try:
                start_time = time.perf_counter()

                documents = retriever(question, k=k)

                end_time = time.perf_counter()

                latency = (end_time - start_time) * 1000

                pages = get_pages(documents)

                precision, recall, mrr = calculate_metrics(
                    pages,
                    expected_pages,
                    k
                )

                results_table.append({
                    "Algorithm": name,
                    "Pages": ", ".join(pages),
                    "Latency": round(latency, 2),
                    "Precision": precision,
                    "Recall": recall,
                    "MRR": mrr
                })

            except Exception as error:

                print(f"\n{name} failed: {error}")

        # Display Comparison Table
        print("\nRETRIEVAL COMPARISON")
        print("-" * 110)

        print(
            f"{'Algorithm':<15}"
            f"{'Pages Retrieved':<30}"
            f"{'Time (ms)':<15}"
            f"{'Precision@4':<15}"
            f"{'Recall@4':<15}"
            f"{'MRR':<10}"
        )

        print("-" * 110)

        for result in results_table:

            precision = (
                f"{result['Precision']:.2f}"
                if result["Precision"] is not None
                else "N/A"
            )

            recall = (
                f"{result['Recall']:.2f}"
                if result["Recall"] is not None
                else "N/A"
            )

            mrr = (
                f"{result['MRR']:.2f}"
                if result["MRR"] is not None
                else "N/A"
            )

            print(
                f"{result['Algorithm']:<15}"
                f"{result['Pages']:<30}"
                f"{result['Latency']:<15}"
                f"{precision:<15}"
                f"{recall:<15}"
                f"{mrr:<10}"
            )

        print("-" * 110)


if __name__ == "__main__":
    evaluate_retrievers(test_cases, k=4)
