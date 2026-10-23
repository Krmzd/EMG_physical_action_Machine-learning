"""A/B test: send the 77 test windows to the endpoint and compare the two variants."""
import json
import math
from collections import defaultdict

import boto3

ENDPOINT = "emg-ab-endpoint"
runtime = boto3.client("sagemaker-runtime", region_name="us-west-2")

with open("ab_payloads.json") as f:
    samples = json.load(f)

results = defaultdict(lambda: {"requests": 0, "correct": 0})

for sample in samples:
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="application/json",
        Body=json.dumps({"features": sample["features"]}),
    )
    variant = response["InvokedProductionVariant"]   # which model answered
    prediction = json.loads(response["Body"].read())
    results[variant]["requests"] += 1
    results[variant]["correct"] += int(prediction["activity_id"] == sample["true"])

for variant in sorted(results):
    r = results[variant]
    print(f"{variant}: {r['requests']} requests, {r['correct']} correct, "
          f"accuracy {r['correct'] / r['requests']:.1%}")

# Is the difference real, or could it be luck? (two-proportion z-test)
a, b = results["VariantA"], results["VariantB"]
p_a, p_b = a["correct"] / a["requests"], b["correct"] / b["requests"]
p_all = (a["correct"] + b["correct"]) / (a["requests"] + b["requests"])
se = math.sqrt(p_all * (1 - p_all) * (1 / a["requests"] + 1 / b["requests"]))
if se > 0:
    z = (p_b - p_a) / se
    p_value = math.erfc(abs(z) / math.sqrt(2))
    print(f"Difference (B - A): {p_b - p_a:+.1%}, p-value: {p_value:.3f}")
else:
    print("Both variants got every request right; no difference to test.")
