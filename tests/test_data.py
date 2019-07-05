import os
import pandas as pd
import numpy as np
import uproot
import h5py
from twaml.data import dataset
from twaml.data import scale_weight_sum
from twaml.data import from_root, from_pytables, from_h5

branches = ["pT_lep1", "pT_lep2", "eta_lep1", "eta_lep2"]
ds = from_root(
    ["tests/data/test_file.root"], name="myds", branches=branches, TeXlabel=r"$t\bar{t}$"
)


def test_name():
    assert ds.name == "myds"
    assert ds.TeXlabel == "$t\\bar{t}$"


def test_no_name():
    dst = from_root(["tests/data/test_file.root"], branches=branches)
    assert dst.name == "test_file.root"


def test_content():
    ts = [uproot.open(f)[ds.tree_name] for f in ds.files]
    raws = [t.array("pT_lep1") for t in ts]
    raw = np.concatenate([raws])
    bins = np.linspace(0, 800, 21)
    n1, bins1 = np.histogram(raw, bins=bins)
    n2, bins2 = np.histogram(ds.df.pT_lep1.to_numpy(), bins=bins)
    np.testing.assert_array_equal(n1, n2)


def test_nothing():
    dst = from_root(["tests/data/test_file.root"], branches=branches)
    assert dst.files[0].exists()


def test_with_executor():
    lds = from_root(["tests/data/test_file.root"], branches=branches, nthreads=4)
    np.testing.assert_array_almost_equal(lds.weights, ds.weights, 8)


def test_weight():
    ts = [uproot.open(f)[ds.tree_name] for f in ds.files]
    raws = [t.array("weight_nominal") for t in ts]
    raw = np.concatenate(raws)
    raw = raw * 150.0
    ds.weights = ds.weights * 150.0
    np.testing.assert_array_almost_equal(raw, ds.weights, 6)


def test_add():
    ds2 = from_root(["tests/data/test_file.root"], name="ds2", branches=branches)
    ds2.weights = ds2.weights * 22
    combined = ds + ds2
    comb_w = np.concatenate([ds.weights, ds2.weights])
    comb_df = pd.concat([ds.df, ds2.df])
    np.testing.assert_array_almost_equal(comb_w, combined.weights, 5)
    np.testing.assert_array_almost_equal(comb_df.get_values(), combined.df.get_values(), 5)
    assert ds.name == combined.name
    assert ds.tree_name == combined.tree_name
    assert ds.label == combined.label


def test_selection():
    ds2 = from_root(
        ["tests/data/test_file.root"],
        name="ds2",
        selection="(reg2j2b==True) & (OS == True) & (pT_lep1 > 50)",
    )
    upt = uproot.open("tests/data/test_file.root")["WtLoop_nominal"]
    reg2j2b = upt.array("reg2j2b")
    OS = upt.array("OS")
    pT_lep1 = upt.array("pT_lep1")
    sel = np.logical_and(np.logical_and(reg2j2b, OS), pT_lep1 > 50)
    w = upt.array("weight_nominal")[sel]
    assert np.allclose(w, ds2.weights)
    # np.testing.assert_array_almost_equal(w, ds2.weights)


def test_append():
    branches = ["pT_lep1", "pT_lep2", "eta_lep1", "eta_lep2"]
    ds1 = from_root(["tests/data/test_file.root"], name="myds", branches=branches)
    ds2 = from_root(["tests/data/test_file.root"], name="ds2", branches=branches)
    ds2.weights = ds2.weights * 5
    # raw
    comb_w = np.concatenate([ds1.weights, ds2.weights])
    comb_df = pd.concat([ds1.df, ds2.df])
    # appended
    ds1.append(ds2)
    # now test
    np.testing.assert_array_almost_equal(comb_w, ds1.weights, 5)
    np.testing.assert_array_almost_equal(comb_df.get_values(), ds1.df.get_values(), 5)


def test_auxweights():
    branches = ["pT_lep1", "pT_lep2", "eta_lep1", "eta_lep2"]
    ds1 = from_root(
        ["tests/data/test_file.root"],
        name="myds",
        branches=branches,
        auxweights=["phi_lep1", "phi_lep2"],
    )
    ds2 = from_root(
        ["tests/data/test_file.root"],
        name="ds2",
        branches=branches,
        auxweights=["phi_lep1", "phi_lep2"],
    )
    ds1.append(ds2)

    dsa = from_root(
        ["tests/data/test_file.root"],
        name="myds",
        branches=branches,
        auxweights=["phi_lep1", "phi_lep2"],
    )
    dsb = from_root(
        ["tests/data/test_file.root"],
        name="ds2",
        branches=branches,
        auxweights=["phi_lep1", "phi_lep2"],
    )
    dsc = dsa + dsb

    np.testing.assert_array_almost_equal(
        ds1.auxweights["phi_lep1"], dsc.auxweights["phi_lep1"], 5
    )

    dsc.change_weights("phi_lep2")
    assert dsc.weight_name == "phi_lep2"

    pl2 = uproot.open("tests/data/test_file.root")["WtLoop_nominal"].array("phi_lep2")
    nw2 = uproot.open("tests/data/test_file.root")["WtLoop_nominal"].array("weight_nominal")
    ds2.change_weights("phi_lep2")
    np.testing.assert_array_almost_equal(ds2.weights, pl2, 5)
    assert "phi_lep2" not in ds2.auxweights
    assert "weight_nominal" in ds2.auxweights

    ds2.to_pytables("outfile1.h5")
    ds2pt = from_pytables("outfile1.h5", "ds2", weight_name="phi_lep2")
    print(ds2pt.auxweights)
    np.testing.assert_array_almost_equal(ds2pt.auxweights["weight_nominal"].to_numpy(), nw2)
    os.remove("outfile1.h5")
    assert True


def test_label():
    ds2 = from_root(["tests/data/test_file.root"], name="ds2", branches=branches)
    assert ds2.label is None
    assert ds2.label_asarray() is None
    ds2.label = 6
    la = ds2.label_asarray()
    la_raw = np.ones_like(ds2.weights, dtype=np.int64) * 6
    np.testing.assert_array_equal(la, la_raw)


def test_auxlabel():
    ds2 = from_root(["tests/data/test_file.root"], name="ds2", branches=branches)
    assert ds2.auxlabel is None
    assert ds2.auxlabel_asarray() is None
    ds2.auxlabel = 3
    assert ds2.auxlabel == 3
    la = ds2.auxlabel_asarray()
    la_raw = np.ones_like(ds2.weights, dtype=np.int64) * 3
    np.testing.assert_array_equal(la, la_raw)


def test_save_and_read():
    ds.to_pytables("outfile.h5")
    new_ds = from_pytables("outfile.h5", ds.name)
    X1 = ds.df.to_numpy()
    X2 = new_ds.df.to_numpy()
    w1 = ds.weights
    w2 = new_ds.weights
    np.testing.assert_array_almost_equal(X1, X2, 6)
    np.testing.assert_array_almost_equal(w1, w2, 6)


def test_raw_h5():
    inds = from_h5("tests/data/raw.h5", "WtLoop_nominal", ["pT_jet1", "nbjets", "met"])
    rawf = h5py.File("tests/data/raw.h5")["WtLoop_nominal"]
    raww = rawf["weight_nominal"]
    rawm = rawf["met"]
    np.testing.assert_array_almost_equal(raww, inds.weights, 5)
    np.testing.assert_array_almost_equal(rawm, inds.df.met, 5)


def test_scale_weight_sum():
    ds1 = from_root(["tests/data/test_file.root"], name="myds", branches=branches)
    ds2 = from_root(["tests/data/test_file.root"], name="ds2", branches=branches)
    ds2.weights = np.random.randn(len(ds1)) * 10
    scale_weight_sum(ds1, ds2)
    testval = abs(1.0 - ds2.weights.sum() / ds1.weights.sum())
    assert testval < 1.0e-4


def test_cleanup():
    os.remove("outfile.h5")
    assert True


def test_columnkeeping():
    ds1 = from_root(
        ["tests/data/test_file.root"],
        name="myds",
        branches=["met", "sumet", "pT_jet2", "reg2j2b"],
        auxweights=["pT_lep1", "pT_lep2", "pT_jet1"],
    )
    keep_c = ["reg2j2b", "pT_jet2"]
    keep_w = ["pT_lep1", "pT_jet1"]
    ds1.keep_columns(keep_c)
    ds1.keep_weights(keep_w)
    list_of_col = list(ds1.df.columns)
    list_of_exw = list(ds1.auxweights.columns)
    assert keep_c == list_of_col
    assert keep_w == list_of_exw


def test_columnrming():
    ds1 = from_root(
        ["tests/data/test_file.root"],
        name="myds",
        branches=["met", "sumet", "pT_jet2", "reg2j2b"],
        auxweights=["pT_lep1", "pT_lep2", "pT_jet1"],
    )

    ds1.rm_columns(["met", "sumet"])
    list_of_cols = list(ds1.df.columns)
    assert len(list_of_cols) == 2 and "pT_jet2" in list_of_cols and "reg2j2b" in list_of_cols

    ds1 = from_root(["tests/data/test_file.root"], name="myds")
    list_of_cols = list(ds1.df.columns)
    assert "OS" in list_of_cols
    assert "SS" in list_of_cols
    assert "elmu" in list_of_cols
    assert "elel" in list_of_cols
    assert "mumu" in list_of_cols
    list_of_regs = [reg for reg in list_of_cols if "reg" in reg]
    ds1.rm_chargeflavor_columns()
    ds1.rm_region_columns()
    ds1.rm_weight_columns()
    list_of_cols_after = list(ds1.df.columns)
    assert "OS" not in list_of_cols_after
    assert "SS" not in list_of_cols_after
    assert "elmu" not in list_of_cols_after
    assert "mumu" not in list_of_cols_after
    assert "elel" not in list_of_cols_after
    assert "reg1j1b" not in list_of_cols_after
    for r in list_of_regs:
        assert r not in list_of_cols_after


def test_selected_datasets():
    ds2 = from_root(
        "tests/data/test_file.root", auxweights=["pT_lep1", "pT_lep2", "pT_jet1"], name="myds"
    )

    splits = ds2.selected_datasets(
        {"s1": "(pT_lep2 > 30) & (pT_jet1 < 50)", "s2": "(reg2j1b==True)"}
    )

    t = uproot.open("tests/data/test_file.root")["WtLoop_nominal"]
    pT_lep2_g30 = t.array("pT_lep2") > 30
    pT_jet1_l50 = t.array("pT_jet1") < 50
    reg2j1b_ist = t.array("reg2j1b") == True

    pT_lep1 = t.array("pT_lep1")
    s1_pT_lep1 = splits["s1"].df.pT_lep1.to_numpy()
    s2_pT_lep1 = splits["s2"].df.pT_lep1.to_numpy()

    pT_lep1_manual_s1 = pT_lep1[pT_lep2_g30 & pT_jet1_l50]
    pT_lep1_manual_s2 = pT_lep1[reg2j1b_ist]

    np.testing.assert_allclose(s1_pT_lep1, pT_lep1_manual_s1)
    np.testing.assert_allclose(s2_pT_lep1, pT_lep1_manual_s2)


def test_selection_masks():
    ds2 = from_root(
        "tests/data/test_file.root",
        auxweights=["pT_lep1", "pT_lep2", "pT_jet1"],
        name="myds",
    )

    masks, sels = ds2.selection_masks(
        {"s1": "(pT_lep2 > 30) & (pT_jet1 < 50)", "s2": "(reg2j1b==True)"}
    )

    t = uproot.open("tests/data/test_file.root")["WtLoop_nominal"]
    pT_lep2_g30 = t.array("pT_lep2") > 30
    pT_jet1_l50 = t.array("pT_jet1") < 50
    reg2j1b_ist = t.array("reg2j1b") == True

    pT_lep1 = t.array("pT_lep1")
    s1_pT_lep1 = ds2[masks["s1"]].df.pT_lep1.to_numpy()
    s2_pT_lep1 = ds2[masks["s2"]].df.pT_lep1.to_numpy()

    pT_lep1_manual_s1 = pT_lep1[pT_lep2_g30 & pT_jet1_l50]
    pT_lep1_manual_s2 = pT_lep1[reg2j1b_ist]

    np.testing.assert_allclose(s1_pT_lep1, pT_lep1_manual_s1)
    np.testing.assert_allclose(s2_pT_lep1, pT_lep1_manual_s2)
