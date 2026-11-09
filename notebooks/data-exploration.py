import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import matplotlib.pyplot as plt
    import polars as pl

    return pl, plt


@app.cell
def _(pl):
    voltage_df = pl.read_csv("../data/pardogomez2026_fig4_voltage_experimental.csv")
    voltage_df
    return (voltage_df,)


@app.cell
def _(pl):
    cod_df = pl.read_csv("../data/pardogomez2026_fig5_cod_experimental.csv")
    cod_df
    return (cod_df,)


@app.cell
def _(cod_df, pl, plt, voltage_df):
    fig, axs = plt.subplots(1, 2, figsize=(24, 6), sharex=True)
    for i in range(4):
        v_df_temp = voltage_df.filter(pl.col("cycle") == i + 1)
        c_df_temp = cod_df.filter(pl.col("cycle") == i + 1)

        axs[0].scatter(v_df_temp["time_d"], v_df_temp["voltage_V"])
        axs[1].scatter(
            c_df_temp["time_d"], c_df_temp["cod_mg_per_L"], label=f"Cycle {i + 1}"
        )

    axs[0].set_ylabel("Voltage (V)")
    axs[0].set_xlabel("Time (d)")

    axs[1].set_ylabel("COD (mg/L)")
    axs[1].set_xlabel("Time (d)")
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
