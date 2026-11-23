import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import trainer

st.set_page_config(page_title="City Hospital | Live Training", page_icon="🏥", layout="wide")
st.title("🏥 City Hospital | Live Training Dashboard")
st.caption("Educational breast-cancer classification demo using scikit-learn's Wisconsin dataset.")
with st.sidebar:
    st.header("Experiment settings")
    hidden = st.selectbox("Hidden neurons", [16, 32, 64, 128, 256], index=0)
    dropout = st.slider("Dropout", 0.0, 0.7, 0.3, 0.05)
    decay = st.select_slider("Weight decay", options=[0.0, 0.001, 0.01, 0.05, 0.1], value=0.05)
    lr = st.select_slider("Learning rate", options=[0.0001, 0.0005, 0.001, 0.003, 0.01], value=0.001)
    epochs = st.slider("Maximum epochs", 20, 300, 160, 10)
    patience = st.slider("Early-stopping patience (0 = off)", 0, 50, 20, 5)
    scale = st.checkbox("Scale features (fit on training data only)", value=True)
    threshold = st.slider("Cancer alert threshold", 0.1, 0.9, 0.5, 0.05)
    run_btn = st.button("▶ Train model", type="primary", use_container_width=True)

if "runs" not in st.session_state: st.session_state.runs = []
if "last" not in st.session_state: st.session_state.last = None

if run_btn:
    cfg = {"hidden": hidden, "dropout": dropout, "weight_decay": decay,
           "lr": lr, "epochs": epochs, "patience": patience,
           "scale": scale, "seed": 0}
    data = trainer.load_data(scale=scale)
    chart_slot = st.empty()
    progress = st.progress(0.0, text="Training model...")
    status = st.empty()
    def redraw(epoch, hist):
        if epoch % 5 == 0 or epoch == 1:
            with chart_slot.container():
                st.subheader("Live learning curves")
                frame = pd.DataFrame({"Training loss": hist["train"],
                                      "Validation loss": hist["val"]})
                st.line_chart(frame)
            progress.progress(min(epoch / epochs, 1.0), text=f"Epoch {epoch}/{epochs}")
            status.caption(f"Current validation loss: {hist['val'][-1]:.4f}")
    model, hist, summary = trainer.train(cfg, data, on_epoch=redraw)
    result = trainer.evaluate(model, data[4], data[5], threshold)
    record = {**cfg, **summary, "test_accuracy": result["accuracy"],
              "cancers_found": result["cancers_found"],
              "false_alarms": result["false_alarms"]}
    st.session_state.runs.append(record)
    st.session_state.last = {"model": model, "hist": hist, "data": data,
                             "result": result, "cfg": cfg, "summary": summary}
    progress.progress(1.0, text="Training complete")
    st.success(f"Experiment saved. Diagnosis: {summary['diagnosis']}")

last = st.session_state.last
if last:
    s, r = last["summary"], last["result"]
    st.subheader("Latest experiment")
    a,b,c,d = st.columns(4)
    a.metric("Diagnosis", s["diagnosis"])
    b.metric("Epochs run", s["epochs_run"])
    c.metric("Best validation loss", f"{s['best_val_loss']:.4f}")
    d.metric("Test accuracy", f"{r['accuracy']:.1%}")
    st.write(f"**Test split screening:** cancers found **{r['cancers_found']}**, false alarms **{r['false_alarms']}**.")
    st.line_chart(pd.DataFrame({"Training loss": last["hist"]["train"],
                                "Validation loss": last["hist"]["val"]}))
    st.subheader("Check a held-out patient")
    Xtest, ytest = last["data"][4], last["data"][5]
    idx = st.number_input("Test patient index", 0, len(ytest)-1, 0)
    patient = trainer.evaluate(last["model"], Xtest[int(idx):int(idx)+1],
                               ytest[int(idx):int(idx)+1], threshold)
    actual = "Malignant" if int(ytest[int(idx)]) == 1 else "Benign"
    predicted = "Malignant" if patient["predictions"][0] == 1 else "Benign"
    st.info(f"Model prediction: **{predicted}** | Reference label: **{actual}** | Alert probability: **{patient['probabilities'][0]:.1%}**")
    st.caption("This is a held-out dataset example, not a clinical diagnosis.")
else:
    st.info("Choose settings in the sidebar and click Train model. Each run is retained below.")

st.subheader("Experiment history")
if st.session_state.runs:
    table = pd.DataFrame(st.session_state.runs)
    st.dataframe(table, use_container_width=True)
    st.download_button("⬇ Download experiments.csv",
        table.to_csv(index=False).encode("utf-8"),
        file_name="experiments.csv", mime="text/csv")
else:
    st.caption("Your experiment table will appear here after the first run.")

st.divider()
st.subheader("Interpretation guide")
st.markdown("""
- **Underfitting:** training loss ends above 0.30.
- **Overfitting:** final validation loss is more than 30% above its best value.
- **Healthy:** neither rule is triggered.
- Compare at least four runs, changing one setting at a time. Prefer validation performance, then evaluate the test set once.
""")
st.warning("Research/teaching demonstration only. Not validated for clinical use. A qualified clinician must review all patient information and decisions.")
