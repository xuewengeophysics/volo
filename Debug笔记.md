# Debug笔记



### load the pretrained weights

+ [There is a problem when loading the pretrained weights](https://github.com/sail-sg/volo/issues/2)
  + when you load pretrain models, don't untar the download file as torch.load() can direclty load '.pth.tar'.



### args.input_size

+ volo

  ```python
  parser.add_argument('--img-size', type=int, default=None, metavar='N',
                      help='Image patch size (default: None => model default)')
  parser.add_argument('--input-size', default=None, nargs=3, type=int,
                      metavar='N N N',
                      help='Input all image dimensions (d h w, e.g. --input-size 3 224 224),'
                           ' uses model default if empty')
  ```

  

+ visformer

  ```python
  parser.add_argument('--input-size', default=224, type=int, help='images input size')
  ```



#### 解决方案：

+ 将datasets.py中用到的`args.input_size`替换为`args.img_size`；





### AttributeError: module 'utils' has no attribute 'init_distributed_mode'

```shell
Traceback (most recent call last):
  File "main_train_custom.py", line 388, in <module>
    main(args)
  File "main_train_custom.py", line 171, in main
    utils.init_distributed_mode(args)
AttributeError: module 'utils' has no attribute 'init_distributed_mode'
```

#### 解决方案：

+ 将`utils.py`中重命名为`utils_custom.py`，将`main_train_custom.py`中用到的`import utils`替换为`import utils_custom as utils`；



### AttributeError: module 'utils' has no attribute 'MetricLogger'

```shell
Traceback (most recent call last):
  File "main_train_custom.py", line 404, in <module>
    main(args)
  File "main_train_custom.py", line 306, in main
    if args.local_rank == 0:
  File "D:\AI\xwStudy\xwGithub\Transformer\volo\engine.py", line 17, in train_one_epoch
    metric_logger = utils.MetricLogger(delimiter="  ")
AttributeError: module 'utils' has no attribute 'MetricLogger'
```

#### 解决方案：

+ 将`engine.py`中用到的`import utils`替换为`import utils_custom as utils`；



### AttributeError: 'tuple' object has no attribute 'log_softmax'

```shell
Traceback (most recent call last):
  File "main_train_custom.py", line 404, in <module>
    main(args)
  File "main_train_custom.py", line 361, in main
    train_stats = train_one_epoch(
  File "D:\AI\xwStudy\xwGithub\Transformer\volo\engine.py", line 30, in train_one_epoch
    loss = criterion(outputs, targets)
  File "D:\Anaconda3\envs\yolov5py38\lib\site-packages\torch\nn\modules\module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "D:\Anaconda3\envs\yolov5py38\lib\site-packages\timm\loss\cross_entropy.py", line 35, in forward
    loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
  File "D:\Anaconda3\envs\yolov5py38\lib\site-packages\torch\nn\functional.py", line 1605, in log_softmax
    ret = input.log_softmax(dim)
AttributeError: 'tuple' object has no attribute 'log_softmax'
```

#### 问题原因：

```python
#--return_dense: use token labeling, details are here: https://github.com/zihangJiang/TokenLabeling
if not self.return_dense:
	return x_cls
else:
	return x_cls, x_aux, (bbx1, bby1, bbx2, bby2)
```

#### 解决办法：

+ 将`volo.py`中的`self.return_dense`设置为 `False`；

  ```python
  self.return_dense = False
  if not self.return_dense:
  	return x_cls
  ```

  

### RuntimeError: invalid argument 5: k not in range for dimension

```shell
Traceback (most recent call last):
  File "main_train_custom.py", line 404, in <module>
    main(args)
  File "main_train_custom.py", line 380, in main
    test_stats = evaluate(data_loader_val, model, device)
  File "D:\Anaconda3\envs\yolov5py38\lib\site-packages\torch\autograd\grad_mode.py", line 26, in decorate_context
    return func(*args, **kwargs)
  File "D:\AI\xwStudy\xwGithub\Transformer\volo\engine.py", line 68, in evaluate
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
  File "D:\Anaconda3\envs\yolov5py38\lib\site-packages\timm\utils\metrics.py", line 29, in accuracy
    _, pred = output.topk(maxk, 1, True, True)
RuntimeError: invalid argument 5: k not in range for dimension at C:/cb/pytorch_1000000000000/work/aten/src\THC/generic/THCTensorTopK.cu:26
```

#### 问题原因：

+ 数据集类别是3类，topk不能取到5；

#### 解决办法：

将`engine.py`中的`topk`设置为 `(1, 2)`；

```python
acc1, acc5 = accuracy(output, target, topk=(1, 2))
```

