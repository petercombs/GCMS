rule all:
    input:
        "sec_CRISPR_vs_WT.tsv",
        "mel_eloF_vs_WT.tsv",
        "mel_eloF_vs_GFP.tsv",
        "mel_bond_vs_WT.tsv",
        "mel_eloF_vs_noRNAi.tsv"


rule figure_2:
    input:
        code="Analysis20170810.py"
    output:
        "pcs.clean.eps",
        "pca_unlabelled.eps",
        "pca_legend.eps",
        "eloF_vs_WT.eps",
        "eloF_sec_vs_WT.eps",
    shell:"""
        python {input.code}
    """

rule crispr_gcms:
    input:
        code="GCMSTests.py"
    output:
        table="sec_CRISPR_vs_WT.tsv",
        plot="sec_CRISPR_vs_WT.eps",
    shell:"""
    python {input.code} \
        -O {output.table} \
        --control-types sechellia_WT \
        --test-types secCRISPR sechellia_CRISPR \
        --norm-peak-start 1060 --norm-peak-stop 1070 \
        --test-label "D.sec eloF-" --control-label "D.sec WT" \
        --test-color "salmon" --control-color "red" \
        -- */*.CDF
        """

rule mel_eloF_noRNAi_gcms:
    input:
        code="GCMSTests.py"
    output:
        table="mel_eloF_vs_noRNAi.tsv",
        plot="mel_eloF_vs_noRNAi.eps",
    shell:"""
    python {input.code} \
        -O {output.table} \
        --control-types eloF947+ gfp186+ bond+ \
        --test-types eloF947- \
        --test-label "D.mel eloF-" --control-label "D.mel WT" \
        -- */*.CDF
        """

rule mel_eloF_gfp_gcms:
    input:
        code="GCMSTests.py"
    output:
        table="mel_eloF_vs_GFP.tsv",
        plot="mel_eloF_vs_GFP.eps",
    shell:"""
    python {input.code} \
        -O {output.table} \
        --control-types gfp201 gfp186- gfp202 \
        --test-types eloF947- \
        --test-label "D.mel eloF-" --control-label "D.mel WT" \
        -- */*.CDF
        """

rule mel_eloF_gcms:
    input:
        code="GCMSTests.py"
    output:
        table="mel_eloF_vs_WT.tsv",
        plot="mel_eloF_vs_WT.eps",
    shell:"""
    python {input.code} \
        -O {output.table} \
        --control-types eloF947+ gfp201- gfp186+ \
        --test-types eloF947- \
        --test-label "D.mel eloF-" --control-label "D.mel WT" \
        -- */*.CDF
        """

rule mel_bond_gcms:
    input:
        code="GCMSTests.py"
    output:
        table="mel_bond_vs_WT.tsv",
        plot="mel_bond_vs_WT.eps",
    shell:"""
    python {input.code} \
        -O {output.table} \
        --control-types bond+ gfp186+ eloF947+ \
        --test-types bond- bond\
        --test-label "D.mel bond-" --control-label "D.mel WT" \
        -- */*.CDF
        """

