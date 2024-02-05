# Lab 3

## Explore additional metrics that can serve as quality metrics for the search process. 

Latency:

We calculate the latency for each input-batch, then accumulate all latencies to take the average for each search option.

<pre>
for i, config in enumerate(search_spaces):
    mg, _ = quantize_transform_pass(mg, config)
    ''''''
    for inputs in data_module.train_dataloader():
        xs, ys = inputs
        start_time = time.time() # start time
        preds = mg.model(xs)
        end_time = time.time() # end time
        latency = end_time - start_time # latency
        latencies.append(latency)
        ''''''
    latency_avg = sum(latencies) / len(latencies) 
</pre>

