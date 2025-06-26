def main():
    from pprint import pprint
    from graph_builder import build_graph
    from vector_db import check_collection

    check_collection()

    app = build_graph()
    # inputs = {"question": "Jakie są wskazania do stosowania leku w postaci kapsułki żelatynowej?"}
    inputs = {"question": """Jakie są możliwe działania niepożądane leków, zawierających w składzie substancję czynną metoprolol?
        """}
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
    print(value["generation"])


if __name__ == "__main__":
    from time import time
    start_time = time()
    main()
    end_time = time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")