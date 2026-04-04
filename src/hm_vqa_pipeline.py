# Pipeline

# Ingestion
# Layer 1: Frames -> OpenCLIP Embedding -> Store
# Layer 2: Reuser layer 1 embedding -> Cut segment by shift in similarity lower limit -> max pooling/mean_pooling 
# Optional for layer 2 after base pipeline done: Object Detection, OCR, ASR, Captioning.
# Layer 3: Cut segment by higher spikes -> Feed these segments into X-CLIP or other video embedding models/methods

# Tune threshold using heuristic search -> justify 

# Query and QA
# Input: user query -> VLM Model -> Create retrieval

# Retrieval:
# Request types: 
#   Action/Video related events 
#       -> Retrieve 3rd layer first -> Get @3
#       -> Retrieve 2nd layers -> subsample frames
#   Direct object/visual related:
#       -> Retrieve 1st layer -> get @5-10 or add time step constrain, ie, the retrieved has to belong to different layer 2 or layer 3 segments.
#       -> LLM run to verify

# Evaluation plan
# Baseline: Subsample 32, 64, 128, 256
# JCEF: Using OpenCLIP setup
# Ours