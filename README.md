# CORRECT
This is the pytorch implementation of NAACL-2025 paper "[CORRECT: Context- and Reference-Augmented Reasoning and Prompting for Scientific Claim Verification](/paper/NAACL25-CORRECT.pdf)", authored by [Delvin Ce Zhang](http://delvincezhang.com/) and [Dongwon Lee](https://pike.psu.edu/dongwon/).

CORRECT is a language model for scientific claim verification task, which aims to verify a given claim using reliable evidence. In this paper, we discover that evidence sentences usually contain insufficient information, and we use auxiliary contextual document and referential documents of evidence sentences to complement them, so that we can verify the claim more accurately.

__Please note that our model can still run without error even with no contexts or references__ by simply removing contextual graph layer and referential graph layer.

![](/paper/figure.jpg)

## Implementation Environment
- python == 3.10
- pytorch == 2.4.1
- transformers == 4.46.0
- numpy == 1.24.1
- sklearn == 1.3.2

## Run
`python main.py -es gold`  # claim verification with gold evidence sentences

`python main.py -es retrieved`   # claim verification with BM25 retrieved evidence sentences

### Parameter Setting
- -dn: dataset name, default = check_covid (choices = \[check_covid, bear_fact, scifact, feverous\])
- -m: mode, default = supervised (choices = \[supervised, few_shot, test\])
- -ns: number of shots for few-shot training, used only when mode == few_shot
- -es: evidence setting, default = gold (choices = \[gold, retrieved\])
- -np: number of prompt embeddings, default = 8
- -rp: whether use random initialization for prompt embeddings, default = False
- -lm: the specific version of BERT language model, default = pubmed_bert (choices = \[base_bert, sci_bert, pubmed_bert, large_bert, multilingual_bert\])
- -ne: number of training epochs, default = 50
- -ls: number of epochs to print and save the results, default = 5
- -lr_lm: learning rate for BERT language model, default = 1e-5
- -lr_p: learning rate for prompt embeddings, default = 1e-3
- -ms: minibatch size, default = 4
- -ml: maximum length of text for tokenization, default = 256
- -n_evid: number of sampled evidence sentences for training, default = 3
- -n_ref: number of sampled referential documents for training, default = 5
- -ddp: whether use distributed training, default = False
- -gpu: gpu
- -rs: random seed

## Data
We release BearFact, Check-COVID, and SciFact datasets in `data.zip` file. Please unzip `data.zip` and put the unzipped data into `./data` folder (e.g., `./data/check_covid/***.json`). For the largest FEVEROUS-S dataset, please email Delvin Ce Zhang (delvincezhang@gmail.com) for access.

Each dataset contains `claims.json`, `evidence.json`, `contexts.json` (__optional__), and `references.json` (__optional__).

Below is an example of `claims.json` format. It is a list, and each element in the list is a dictionary containing information of a specific claim. The length of the list is the number of total claims.

```
[
    {
        "claim_id": 0,  # claim id (may not start from 0, may not be an integer)
        "claim_text": "One type of COVID-19 test identifies coronavirus proteins in a few seconds.",  # claim text (a string)
        "label": "refute",  # label (can be any string, may not strictly be support, refute, or nei)
        "train_dev_test": "train",  # dataset split (optional. If provided with train, dev, or test, the model will follow the split. If not provided, the model will split the data into 72:8:20 for train:dev:test)
        "gold_evid_ids": [  # a list of gold evidence ids (used only when args.evidence_setting == gold, these evidence ids correspond to the ids in evidences.json)
            0
        ],
        "bm25_retrieved_evid_ids": [  # a list of retrieved evidence ids (used only when args.evidence_setting == retrieved)
            1161,
            1308,
            465
        ]
    },
    {
        "claim_id": 1,
        "claim_text": "procalcitonin have/has a positive influence on COVID-19",
        "label": "support",
        "gold_evid_ids": [
            1
        ],
        "bm25_retrieved_evid_ids": [
            1358,
            641,
            765
        ]
    }
]
```

Below is an example of `evidence.json` format. It is a list, and each element in the list is a dictionary containing information of a specific evidence sentence. The length of the list is the number of total evidence sentences.

```
[
    {
        "evid_id": 0,  # evidence id (this id corresponds to gold_evid_ids and bm25_retrieved_evid_ids in claims.json, this id may not start from 0 and may not be an integer)
        "s2orc_id": 219688666,  # s2orc id (optional)
        "title": "Field-deployable, rapid diagnostic testing of saliva samples for SARS-CoV-2.",  # title of the paper that contains this evidence sentence (optional, a string)
        "evid_text": "We developed an assay that detects single copies of SARS-CoV-2 virus directly from saliva and swab samples in 30 min using a simple, one-step protocol that utilizes only a heat block and microcentrifuge tube prefilled with a mixture containing the necessary reagents and has a sensitivity and specificity of 97% and 100%, respectively.",  # evidence text (a string)
        "ctx_id": [  # a list of contextual document id (optional)
            0
        ],
        "ref_ids": [  # a list of referential document ids (optional)
            0,
            1,
            2,
            3,
            4,
            5,
            6
        ]
    },
    {
        "evid_id": 1,
        "s2orc_id": 219688666,
        "title": "Field-deployable, rapid diagnostic testing of saliva samples for SARS-CoV-2.",
        "evid_text": "We developed an assay that detects single copies of SARS-CoV-2 virus directly from saliva and swab samples in 30 min using a simple, one-step protocol that utilizes only a heat block and microcentrifuge tube prefilled with a mixture containing the necessary reagents and has a sensitivity and specificity of 97% and 100%, respectively.",
        "ctx_id": [
            0
        ],
        "ref_ids": [
            0,
            1,
            2,
            3,
            4,
            5,
            6
        ]
    }
]
```

Below is an example of `contexts.json` format. It is a list, and each element in the list is a dictionary containing information of a specific contextual document. The length of the list is the number of total contextual documents. `references.json` has the same format and is not double explained. Both `contexts.json` and `references.json` are optional to provide. If provided, the model will contruct a three-layer graph and use them. If not provided, the model can still run.

```
[
    {
        "ctx_id": 0,  # contextual document id
        "s2orc_id": 219688666,  # s2orc id (optional)
        "title": "Field-deployable, rapid diagnostic testing of saliva samples for SARS-CoV-2.",  # title of the paper that contains this contextual document or abstract (optional, a string)
        "ctx_text": "Abstract Rapid, scalable, point-of-need, COVID-19 diagnostic testing is necessary to safely re-open economies and prevent future outbreaks. We developed an assay that detects single copies of SARS-CoV-2 virus directly from saliva and swab samples in 30 min using a simple, one-step protocol that utilizes only a heat block and microcentrifuge tube prefilled with a mixture containing the necessary reagents and has a sensitivity and specificity of 97% and 100%, respectively."  # the full content of the contextual document (a string)
    },
    {
        "ctx_id": 1,
        "s2orc_id": 263151624,
        "title": "The major genetic risk factor for severe COVID-19 is inherited from Neandertals",
        "ctx_text": "A recent genetic association study (Ellinghaus et al. 2020) identified a gene cluster on chromosome 3 as a risk locus for respiratory failure in SARS-CoV-2. Recent data comprising 3,199 hospitalized COVID-19 patients and controls reproduce this and find that it is the major genetic risk factor for severe SARS-CoV-2 infection and hospitalization (COVID-19 Host Genetics Initiative). Here, we show that the risk is conferred by a genomic segment of ~50 kb that is inherited from Neandertals and occurs at a frequency of ~30% in south Asia and ~8% in Europe."
    }
]
```

## Output
We print verification result (with both Micro F1 score and Macro F1 score) and save model checkpoints every `args.log_steps (default = 5)` epochs. Model checkpoints are saved to the `./ckpt` folder.

## Reference
If you find our paper useful, including code and data, please cite

```
@inproceedings{correct,
    author = {Zhang, Delvin Ce and Lee, Dongwon},
    title = {CORRECT: Context- and Reference-Augmented Reasoning and Prompting for Fact-Checking},
    year = {2025},
    booktitle = {Proceedings of 2025 Annual Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics},
    series = {NAACL '25}
}
```
