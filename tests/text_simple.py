from zeromodel import ZeroModel
import numpy as np

metric_names = ["uncertainty", "size", "quality", "novelty", "coherence"]

def test_zeromodel():

    # 1. Prepare your data (100 documents × 5 metrics)
    score_matrix = np.random.rand(100, 5)

    # 2. Initialize ZeroModel with your metrics
    zeromodel = ZeroModel(metric_names)

    # 3. Set your task (natural language description)
    zeromodel.set_task("Find uncertain large documents")

    # 4. Process your data (creates spatial organization)
    zeromodel.process(score_matrix)

    # 5. Get the top decision in one line
    doc_idx, relevance = zeromodel.get_decision()

    print(f"Top document: #{doc_idx} with relevance {relevance:.2f}")
    # Output: Top document: #7 with relevance 0.92

def test_edge():
    # On a device with <25KB memory:
    zeromodel = ZeroModel(metric_names)

    tile = zeromodel.get_critical_tile()  # Just 9 bytes of data

    # Only 180 bytes of code needed:
    def process_tile(tile_data):
        return tile_data[4] < 128  # Is top-left pixel "dark enough"?

    is_relevant = process_tile(tile)
    print(f"Edge device decision: {'RELEVANT' if is_relevant else 'NOT RELEVANT'}")
    # Output: Edge device decision: RELEVANT