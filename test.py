import evaluate
import importlib
import datasets
import os

metric_name="bertscore"


metric_module = importlib.import_module(
            evaluate.load.metric_module_factory(os.path.join("metrics", metric_name)).module_path
        )

#print(datasets.load.metric_module_factory(os.path.join("metrics", metric_name)).module_path)
#print(metric_module)

print(metric_module.__name__)

metric = evaluate.load.import_main_class(metric_module.__name__, dataset=False)

print()
print(metric)
print()

metric().compute()