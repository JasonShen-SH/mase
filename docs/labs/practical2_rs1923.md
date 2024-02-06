# Lab 3

## 1. Explore additional metrics that can serve as quality metrics for the search process. 

**Latency**: (Unit: ms, we have multiplied by 1000)

For each search option, we calculate the latency for each input-batch, then accumulate all latencies to take the average.
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
        latencies.append(latency*1000)
        ''''''
    latency_avg = sum(latencies) / len(latencies) 
</pre>


**model size**: (Unit: Byte)

It is presupposed that the model, whose size is to be calculated, has already undergone quantization, each time with different search option.

For each search option, we calculate the total storage size of the model by iterating through the space occupied by the weights of each layer.
<pre>
The memory footprint of each layer is determined by the following attributes:
Linear: weight, bias
Batchnorm: weight(γ), bias(β), mean, variance
ReLU: None
</pre>

The subsequent script is designed for assessing the memory consumption attributed to the model
<pre>
def model_storage_size(model, weight_bit_width, bias_bit_width, data_bit_width):
    total_bits = 0 
    for name, param in model.named_parameters():
        if param.requires_grad and 'weight' in name:
            bits = param.numel() * weight_bit_width
            total_bits += bits
        elif param.requires_grad and 'bias' in name:
            bits = param.numel() * bias_bit_width
            total_bits += bits
    total_bits += data_bit_width*(1*16+1) # mean and variance of batchnorm
    total_bytes = total_bits / 8
    return total_bytes

for i, config in enumerate(search_spaces):
    # definition of weight & bias & data width
    size = model_storage_size(mg.model, weight_bit_width, bias_bit_width, data_bit_width)
    ''''''
</pre>


**Bit-wise operations**: (Unit: number)

For each search option, we compute the bitwise operations count for the linear module。

We employ the identical methodology as outlined in the optional task of Lab2.

<pre>
def bit_wise_op(model, input_res, data_width, weight_width, bias_width, batch_size):
    total_bitwise_ops = 0
    for name, module in model.named_modules():
        if isinstance(module, LinearInteger):
            bitwise_ops = calculate_bitwise_ops_for_linear(module, input_res, data_width, weight_width, bias_width, batch_size)
            total_bitwise_ops += bitwise_ops
    return total_bitwise_ops
def calculate_bitwise_ops_for_linear(module, input_res, data_bit_width, weight_bit_width, bias_bit_width, batch_size):
    in_features = module.in_features
    out_features = module.out_features
    bitwise_ops_per_multiplication = data_bit_width * weight_bit_width
    bitwise_ops_per_addition = data_bit_width * weight_bit_width
    bitwise_ops_per_output_feature = in_features * bitwise_ops_per_multiplication + (in_features - 1) * bitwise_ops_per_addition
    if module.bias is not None:
        bitwise_ops_per_output_feature += bias_bit_width
    total_bitwise_ops = out_features * bitwise_ops_per_output_feature
    return total_bitwise_ops*batch_size

for i, config in enumerate(search_spaces):
    # definition of weight & bias & data width
    bit_op = bit_wise_op(mg.model, (16,), data_bit_width, weight_bit_width, bias_bit_width)
    ''''''
</pre>

## 2. Implement some of these additional metrics and attempt to combine them with the accuracy or loss quality metric

<pre>
# Essential Code Segment (Extraneous elements omitted)
for i, config in enumerate(search_spaces):
    size = model_storage_size(mg.model, weight_bit_width, bias_bit_width, data_bit_width)  # model size after it has been quantized
    bit_op = bit_wise_op(mg.model, (16,), weight_bit_width, bias_bit_width, data_bit_width, batch_size)

    acc_avg, loss_avg = 0, 0;  accs, losses, latencies = [], [], []

    for inputs in data_module.train_dataloader():
        xs, ys = inputs
        start_time = time.time()
        preds = mg.model(xs)
        end_time = time.time()
        latency = end_time - start_time
        latencies.append(latency)

        acc = metric(preds, ys); accs.append(acc)
        loss = torch.nn.functional.cross_entropy(preds, ys); losses.append(loss)

    acc_avg = sum(accs) / len(accs)
    loss_avg = sum(losses) / len(losses)
    latency_avg = sum(latencies) / len(latencies) 

    # for this particular element in search space
    recorded_metrics.append({
        ......
    })   
</pre>

<img src="../../imgs/3_2.png" width=500>



## 3. Implement the brute-force search as an additional search method within the system, this would be a new search strategy in MASE.

We could ac

