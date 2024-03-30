import pandas as pd


def _1_match_data() -> None:
    """Make list of samples that appear in all datasets"""
    ge = pd.read_csv("data/input/raw/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena",
                     delimiter="\t").set_index("sample").T
    cn = pd.read_csv("data/input/raw/Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
                     delimiter="\t").set_index("Sample").T
    me = pd.read_csv("data/input/raw/HumanMethylation27",
                     delimiter="\t").set_index("Sample").T
    mi = pd.read_csv("data/input/raw/pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.xena",
                     delimiter="\t").set_index("sample").T
    samples = pd.read_csv("data/input/raw/TCGA_phenotype_denseDataOnlyDownload.tsv",
                          delimiter="\t")

    temp = pd.concat([samples["sample"], me.index.to_series()], copy=False, ignore_index=True)
    ids = temp[temp.duplicated()]
    print(f"A: {ids.shape}", flush=True)

    temp = pd.concat([ids, ge.index.to_series()], copy=False, ignore_index=True)
    ids = temp[temp.duplicated()]
    print(f"B: {ids.shape}", flush=True)

    temp = pd.concat([ids, cn.index.to_series()], copy=False, ignore_index=True)
    ids = temp[temp.duplicated()]
    print(f"C: {ids.shape}", flush=True)

    temp = pd.concat([ids, mi.index.to_series()], copy=False, ignore_index=True)
    ids = temp[temp.duplicated()]
    print(f"D: {ids.shape}", flush=True)

    print(ids.head())
    print(ids.shape)
    ids.to_csv("data/input/ids.csv", index=False)


def _2_filter_samples() -> None:
    """Filter out all samples that don't occur in all datasets"""
    ids: pd.Series = pd.read_csv("data/input/ids.csv")["0"]
    omic_dfs = {}

    omic_dfs["ge"] = pd.read_csv("data/input/raw/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena",
                                 delimiter="\t").set_index("sample").T
    omic_dfs["cn"] = pd.read_csv("data/input/raw/Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
                                 delimiter="\t").set_index("Sample").T
    omic_dfs["me"] = pd.read_csv("data/input/raw/HumanMethylation27",
                                 delimiter="\t").set_index("Sample").T
    omic_dfs["mi"] = pd.read_csv("data/input/raw/pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.xena",
                                 delimiter="\t").set_index("sample").T

    samples = pd.read_csv("data/input/raw/TCGA_phenotype_denseDataOnlyDownload.tsv",
                          delimiter="\t")

    y = samples[samples["sample"].isin(ids)][["sample", "_primary_disease"]]
    y.to_csv("data/input/omics/y.csv", index=False)
    for key in omic_dfs:
        omic_dfs[key] = omic_dfs[key][omic_dfs[key].index.to_series().isin(ids)]
        omic_dfs[key].to_csv(f"data/input/omics/{key}.csv")


def _3_filter_nan_values() -> None:
    """
    Drop any features where values are missing
    ge: before (1439, 20532), after (1439, 16341)
    cn: before (1439, 24777), after (1439, 24777)
    me: before (1439, 27579), after (1439, 19624)
    mi: before (1439,   744), after (1439,   744)

    no_me
    ge: before (9606, 20532), after (9606, 16336)
    cn: before (9606, 24777), after (9606, 24777)
    mi: before (9606, 744), after (9606, 744)
    """

    omic_dfs: dict[str, pd.DataFrame] = {}
    omic_dfs["ge"] = pd.read_csv("data/input/omics/ge.csv")
    omic_dfs["cn"] = pd.read_csv("data/input/omics/cn.csv")
    omic_dfs["me"] = pd.read_csv("data/input/omics/me.csv")
    omic_dfs["mi"] = pd.read_csv("data/input/omics/mi.csv")
    y = pd.read_csv("data/input/omics/y.csv")

    for df_name, df in omic_dfs.items():
        before = df.shape
        omic_dfs[df_name] = df.dropna(axis=1, how="any")
        print(f"{df_name}: before {before}, after {omic_dfs[df_name].shape}")
        omic_dfs[df_name].to_csv(f"data/input/omics/{df_name}.csv")


def _4_class_balance_analysis() -> None:
    """Also adds TCGA abbreviations to the label CSVs"""
    y = pd.read_csv("data/input/omics/y.csv")
    y_no_me = pd.read_csv("data/input/omics/y_no_me.csv")

    print(y.value_counts(subset=["_primary_disease"]))
    print(y_no_me.value_counts(subset=["_primary_disease"]))

    abbrv = pd.read_csv("data/input/raw/tcga_abbreviations.csv")
    abbrv = pd.Series(abbrv["abbreviation"].values, index=abbrv["study_name"]).to_dict()

    y["label"] = y["_primary_disease"].copy()
    y["label"].replace(abbrv, inplace=True)
    y.to_csv("data/input/omics/y.csv")

    y_no_me["label"] = y_no_me["_primary_disease"].copy()
    y_no_me["label"].replace(abbrv, inplace=True)
    y_no_me.to_csv("data/input/omics/y_no_me.csv")


if __name__ == "__main__":
    _4_class_balance_analysis()
