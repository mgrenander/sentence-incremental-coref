import json
import argparse
from tensorize import CorefDataProcessor
from run import Runner
import logging
logging.getLogger().setLevel(logging.CRITICAL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True, help='Configuration name in experiments.conf')
    parser.add_argument('--model_identifier', type=str, required=True, help='Model identifier to load')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output')
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU id; CPU by default')
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--eval_data', type=str, default='dev')
    args = parser.parse_args()

    runner = Runner(args.config_name, args.gpu_id)
    model = runner.initialize_model(args.model_identifier)
    data_processor = CorefDataProcessor(runner.config)
    examples_train, examples_dev, examples_test = runner.data.get_tensor_examples()
    stored_info = runner.data.get_stored_info()
    tensor_examples = examples_dev if args.eval_data == 'dev' else examples_test  # Change for other eval sets

    with open(args.input_path, 'r') as f:
        lines = f.readlines()
    docs = [json.loads(line) for line in lines]
    result = runner.predict(model, tensor_examples)

    with open(args.output_path, 'w') as f:
        for i, doc in enumerate(docs):
            doc['predicted_mention_actions'] = result[i]['predicted_mention_actions']
            doc['predicted_clusters'] = result[i]['predicted_clusters']
            doc['gold_coref_actions'] = result[i]['gold_coref_actions']
            doc['predicted_coref_actions'] = result[i]['predicted_coref_actions']
            f.write(json.dumps(doc) + '\n')
    print(f'Saved prediction in {args.output_path}')
