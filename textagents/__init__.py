"""Multi-agent reading of financial text, and the econometrics that judges whether it helps.

Four steps, in the order they run:

    corpus      step 2  select the documents a (instrument, session) unit may see, and assemble
                        them into the single digest every reader of that unit receives
    chain       step 3  the nine calls: three analysts, two contradictors, one forecaster, three
                        magnitude reviewers, each an independent call on typed inputs
    chain       step 4  the fixed arithmetic rule that reduces those nine objects to three
                        numbers: tone, ambiguity, expected amplitude
    analysis            the nested HAR contrasts and the out-of-sample test that decide whether
                        those three numbers add anything to the price-only reference

`llm` holds the inference client, its cache, its ledger and its spend cap. `regime` holds the
calm / stressed split, defined once. `cli` holds the two entry points: the nine-call chain and
the single-call control it is measured against.
"""

__all__ = ["analysis", "chain", "cli", "config", "corpus", "llm", "regime"]
