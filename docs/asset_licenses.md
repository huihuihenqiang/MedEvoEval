# Asset licenses and API terms

The released 700 case records derive from Chinese MedQA. The upstream repository supplies the MIT license (Copyright 2022 Di Jin); LICENSE_MedQA.txt in the data package preserves that notice. Source: https://github.com/jind11/MedQA ; license: https://github.com/jind11/MedQA/blob/master/LICENSE . The source paper is Di Jin et al., Applied Sciences 11(14):6421 (2021), https://doi.org/10.3390/app11146421 .

Project-added code, schemas, documentation and process derivatives retain the supplied CC BY-NC-SA 4.0 license: https://creativecommons.org/licenses/by-nc-sa/4.0/ . The upstream MIT license continues to apply to the MedQA source text.

The experiment runner accesses hosted models through Alibaba Cloud Model Studio. Qwen and third-party services are subject to the applicable service agreements; this package distributes no model weights. Official agreement index: https://help.aliyun.com/zh/model-studio/related-agreements . Third-party agreement list (including Kimi and GLM): https://terms.alicdn.com/legal-agreement/terms/common_product_agreement/20260207131114217/20260207131114217.html . Kimi terms: https://platform.moonshot.cn/docs/agreement/modeluse .

Model-generated process files are identified as simulated outputs. Reuse of these outputs is also subject to the applicable provider terms. The project license does not replace those terms. 

Software dependencies are installed from the requirements files rather than bundled as vendor source. The dependency notice below records versions and license metadata from the tested environment.

| Dependency | Tested version | License metadata |
|---|---|---|
| requests | 2.32.5 | License :: OSI Approved :: Apache Software License |
| PyYAML | 6.0.3 | License :: OSI Approved :: MIT License |
| scikit-learn | 1.3.0 | License :: OSI Approved :: BSD License |
| tqdm | 4.67.3 | MPL-2.0 AND MIT |
| numpy | 1.26.4 | License :: OSI Approved :: BSD License |
| scipy | 1.15.3 | License :: OSI Approved :: BSD License |
| matplotlib | 3.7.2 | License :: OSI Approved :: Python Software Foundation License |
| jsonschema | 4.26.0 | MIT |
