# Prompts

The eleven files in this folder are the instructions given to the model. Each one is sent as the
system message of a single reader, concatenated with the output schema of that reader's stage, so
a file here fixes what a reader is asked to look for and never the shape of what it returns.

## What each file corresponds to

The reading chain makes nine calls per unit, in four stages.

| File | Stage |
| -- | -- |
| `analyst_fundamentals.txt` | Analyst 1 of 3: accounts and capital structure |
| `analyst_exogeneity_v2.txt` | Analyst 2 of 3: shocks external to the issuer. This is the wording used by the campaign |
| `analyst_exogeneity.txt` | Earlier wording of the same analyst seat, kept because its readings are in the published results. Not the default |
| `analyst_tone.txt` | Analyst 3 of 3: tone of the documents and whether they carry new information |
| `debate_supportive.txt` | Contradictor 1 of 2: argues the favourable case from the three analyst reports |
| `debate_adverse.txt` | Contradictor 2 of 2: argues the unfavourable case from the same reports |
| `forecaster.txt` | The forecaster: settles on a magnitude for the next session, not a direction |
| `panel_central.txt` | Magnitude reviewer 1 of 3: the central case |
| `panel_conservative.txt` | Magnitude reviewer 2 of 3: the subdued case |
| `panel_extreme.txt` | Magnitude reviewer 3 of 3: the high magnitude case |

The control makes one call per unit, on the same document digest.

| File | Stage |
| -- | -- |
| `single_reader.txt` | The control reader: one pass, returning in one object what the chain returns in nine calls |

Filenames are kept as they ran, so `panel_*` names the three magnitude reviewers of the fourth
stage.

## Language

These instructions are in English; the documents they are pointed at are French press material.
Every file that asks for a quotation therefore says the quotation is copied word for word and
stays in the language of the source, so the `evidence` field remains verifiable against the
document it came from.

The campaign reported in the dissertation ran on a French wording of these same instructions.
Translating them changes their fingerprint, and the fingerprint is part of the response cache key,
so a run against the files as they stand here starts from an empty cache and its readings may
differ from the published ones. What each reader is asked to look for is unchanged.

## Fingerprints

Digests of the instruction text with newlines normalised to LF, which is the form the client
hashes and sends. `textagents/llm/client.py` records the first sixteen characters of these next to
every call in the run ledger.

```
analyst_exogeneity.txt      1da30f583c72eda4f37c4a3f54dadaa74274a65c9b488805435243e310c29e2f
analyst_exogeneity_v2.txt   cca4817e3993ffb61cf8ee80c3c322257825880d329e6b6c666c2211cd670a8c
analyst_fundamentals.txt    a86c2b034ba73ff379eb3bdc516072e12f2b1e0fafe7f89bb0e023022cdc77e2
analyst_tone.txt            aced968e5c62c7c756fe91b9c30aec1749544417060450c0275507e52794ea69
debate_adverse.txt          21982224b77b4dfa4c715a347b4b41c2f43a926ef462c655dc88d77058be72e1
debate_supportive.txt       80f7686a200b4db6c4ec130dd506676049678103f1d821960ce36708756a1f44
forecaster.txt              19adb14537921c7ffa05835b3ffe24774fd27818fed054cd923ff79b319b4a61
panel_central.txt           3ddf11d04c75ec75c75f09de0eb7f9ead40abd170b23f8a3f720969847edb92e
panel_conservative.txt      524c09b17b5843c3a752489b00fa5dfe3dc7ad323fcc4ff7ce35f7bf459b6ec3
panel_extreme.txt           1b8da8be4f9817588cf1142119065203e6d7aeaab9fd686e356d2481e27f29f0
single_reader.txt           33358ca4ab204ba9c3b434eeba02d383b44d87b2522de3cb0c8fe93d4fd88d3f
```

Recompute them with:

```bash
python -c "import hashlib,pathlib; [print(f'{p.name:<27}', hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()) for p in sorted(pathlib.Path('textagents/prompts').glob('*.txt'))]"
```
