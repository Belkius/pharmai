def main():
    from pprint import pprint
    from graph_builder import build_graph
    from vector_db import check_collection

    check_collection()


    app = build_graph()
    # inputs = {"question": "Jakie są wskazania do stosowania leku w postaci kapsułki żelatynowej?"}
    inputs = {"question": "Jakie hitler miał włosy?"}
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
    pprint(value["generation"])


if __name__ == "__main__":
    main()
