cd ~/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/GPT4-prompt-experiments
conda activate vlog


OUTPUT_DIR="/home/sushant/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/GPT4-prompt-experiments/benchmark_qa_json_scores"

python ./../quantitative_evaluation/evaluate_benchmark_1_correctness.py \
  --pred_path "benchmark_qa_json/generic_qa_pred.json" \
  --output_dir "${OUTPUT_DIR}/correctness_eval" \
  --output_json "${OUTPUT_DIR}/correctness_results.json" \
  --num_tasks 10

  # Run the "detailed orientation" evaluation script
python  ./../quantitative_evaluation/evaluate_benchmark_2_detailed_orientation.py \
  --pred_path "benchmark_qa_json/generic_qa_pred.json" \
  --output_dir "${OUTPUT_DIR}/detailed_eval" \
  --output_json "${OUTPUT_DIR}/detailed_orientation_results.json" \
  --num_tasks 10



# Run the "contextual understanding" evaluation script
python  ./../quantitative_evaluation/evaluate_benchmark_3_context.py \
  --pred_path "benchmark_qa_json/generic_qa_pred.json" \
  --output_dir "${OUTPUT_DIR}/context_eval" \
  --output_json "${OUTPUT_DIR}/contextual_understanding_results.json" \
  --num_tasks 10


# Run the "temporal understanding" evaluation script
python ./../quantitative_evaluation/evaluate_benchmark_4_temporal.py \
  --pred_path "benchmark_qa_json/temporal_qa_pred.json" \
  --output_dir "${OUTPUT_DIR}/temporal_eval" \
  --output_json "${OUTPUT_DIR}/temporal_understanding_results.json" \
  --num_tasks 10


# Run the "consistency" evaluation script
python ./../quantitative_evaluation/evaluate_benchmark_5_consistency.py \
  --pred_path "benchmark_qa_json/consistency_qa_pred.json" \
  --output_dir "${OUTPUT_DIR}/consistency_eval" \
  --output_json "${OUTPUT_DIR}/consistency_results.json" \
  --num_tasks 10
