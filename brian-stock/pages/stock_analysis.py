def prepare_research_dataset(
    data,
):
    if (
        data is None
        or data.empty
        or "Close" not in data.columns
    ):
        return None

    df = data.copy()

    # ========================================================
    # CHUẨN HÓA NUMERIC
    # ========================================================

    for column in df.columns:

        if (
            column != "Time"
            and column != "Date"
        ):

            try:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

            except Exception:
                pass

    # ========================================================
    # RETURN
    # ========================================================

    if "Return" not in df.columns:

        df["Return"] = (
            df["Close"]
            .pct_change()
        )

    if "ReturnPct" not in df.columns:

        df["ReturnPct"] = (
            df["Return"]
            * 100
        )

    # ========================================================
    # TOÀN BỘ BIẾN SỐ
    #
    # KHÔNG giới hạn 4-6 biến như code cũ.
    # ========================================================

    excluded = {
        "Target_1D",
        "Target_5D",
        "Target_20D",
    }

    features = []

    for column in df.columns:

        if column in excluded:
            continue

        if column in {
            "Time",
            "Date",
        }:
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            features.append(
                column
            )

    # ========================================================
    # GIỮ NGUYÊN THỨ TỰ
    # ========================================================

    # Không sort alphabet để dễ đối chiếu
    # với market.py.
    features = list(
        dict.fromkeys(
            features
        )
    )

    # ========================================================
    # FUTURE TARGETS
    # ========================================================

    close = pd.to_numeric(
        df["Close"],
        errors="coerce",
    )

    df[
        "Target_1D"
    ] = (
        close.shift(-1)
        / close
        - 1
    )

    df[
        "Target_5D"
    ] = (
        close.shift(-5)
        / close
        - 1
    )

    df[
        "Target_20D"
    ] = (
        close.shift(-20)
        / close
        - 1
    )

    # ========================================================
    # LÀM SẠCH
    # ========================================================

    df = df.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    # ========================================================
    # KHÔNG XÓA BIẾN
    #
    # Mỗi biến thiếu dữ liệu sẽ được median-fill.
    # ========================================================

    X = df[
        features
    ].copy()

    for column in features:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

        median = X[
            column
        ].median()

        if pd.isna(
            median
        ):

            median = 0.0

        X[
            column
        ] = X[
            column
        ].fillna(
            median
        )

    # ========================================================
    # DATASET CHO TỪNG HORIZON
    # ========================================================

    datasets = {}

    for horizon in [
        "1D",
        "5D",
        "20D",
    ]:

        target = df[
            f"Target_{horizon}"
        ]

        valid = (
            target.notna()
            & np.isfinite(
                target
            )
        )

        X_h = X.loc[
            valid
        ].copy()

        y_h = target.loc[
            valid
        ].copy()

        if len(X_h) >= 20:

            datasets[
                horizon
            ] = {
                "X": X_h,
                "y": y_h,
            }

    if not datasets:

        return None

    return {
        "features": features,
        "datasets": datasets,
        "rows_raw": len(df),
    }
