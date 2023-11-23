# OneShot-AlphaNet

### Prepare Dataset
Download the Imagenet-100 from the links:
https://sutdapac-my.sharepoint.com/:u:/g/personal/vovan_tuan_sutd_edu_sg/EaWA3oLM575Nv_0mXoL7vlYBlhJ5IZvGc1YbjIjkavovUg?e=e5v6HM

UnZip and Move all downloaded folders into the `./dataset`

## Evaluation Evoluation Search

- Please first download our [pretrained AlphaNet models on Imagenet-100](https://sutdapac-my.sharepoint.com/:u:/g/personal/vovan_tuan_sutd_edu_sg/ETQkn5ltU7lOjSBDaZ7VtxoBRSLZtLfmxGRIbEkZYPM5_Q?e=ZtN1pt)  and put the pretrained models under your local folder *./alphanet_data*


To search the Pareto models for the best FLOPs vs. accuracy tradeoffs in _parallel_supernet_evo_search.py_; to run this example:
```python
python parallel_supernet_evo_search.py --config-file configs/parallel_supernet_evo_search.yml 
```

In case search with fixed some layer, please change the config file to fixed layer: ./configs/parallel_supernet_evo_search.yml

'''python
supernet_config_fix:
    use_v3_head: True
    resolutions: [192, 224, 256, 288]
    first_conv: 
        c: [16]
        act_func: 'swish'
        s: 2
    mb1:
        c: [16]
        d: [1]
        k: [3]
        t: [1]
        s: 1
        act_func: 'swish'
        se: False
    mb2:
        c: [24]
        d: [3]
        k: [3]
        t: [4]
        s: 2
        act_func: 'swish'
        se: False
    mb3:
        c: [32, 40] 
        d: [3, 4, 5, 6]
        k: [3, 5]
        t: [4, 5, 6]
        s: 2
        act_func: 'swish'
        se: True
    mb4:
        c: [64, 72] 
        d: [3, 4, 5, 6]
        k: [3, 5]
        t: [4, 5, 6]
        s: 2
        act_func: 'swish'
        se: False
    mb5:
        c: [112, 120, 128] 
        d: [3, 4, 5, 6, 7, 8]
        k: [3, 5]
        t: [4, 5, 6]
        s: 1
        act_func: 'swish'
        se: True
    mb6:
        c: [192, 200, 208, 216] 
        d: [3, 4, 5, 6, 7, 8]
        k: [3, 5]
        t: [6]
        s: 2
        act_func: 'swish'
        se: True
    mb7:
        c: [216, 224] 
        d: [1, 2]
        k: [3, 5]
        t: [6]
        s: 1
        act_func: 'swish'
        se: True
    last_conv:
        c: [1792, 1984]
        act_func: 'swish'
'''

