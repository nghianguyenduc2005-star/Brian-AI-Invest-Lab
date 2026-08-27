import streamlit as st


def render_sidebar():

    # ========================================================
    # KHỞI TẠO TRANG
    # ========================================================

    if "page" not in st.session_state:
        st.session_state["page"] = "Dashboard"

    # ========================================================
    # GIAO DIỆN THANH BÊN
    # ========================================================

    with st.sidebar:

        st.markdown(
            """
            <div style="
                padding: 8px 4px 22px 4px;
                text-align: left;
            ">
                <div style="
                    font-size: 11px;
                    color: #94a3b8;
                    margin-bottom: 14px;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                ">
                    BRIAN STOCK
                </div>

                <div style="
                    font-size: 24px;
                    font-weight: 800;
                    color: #ffffff;
                    margin-bottom: 5px;
                ">
                    📊 BRIAN STOCK
                </div>

                <div style="
                    font-size: 12px;
                    color: #94a3b8;
                ">
                    Investment Research Platform
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "---"
        )

        # ====================================================
        # HÀM CHUYỂN TRANG
        # ====================================================

        def _chuyen_trang(
            ten_trang,
        ):

            st.session_state[
                "page"
            ] = ten_trang

        # ====================================================
        # DASHBOARD
        # ====================================================

        if st.button(
            "📊 Dashboard",
            key="nut_dashboard",
            width="stretch",
            type=(
                "primary"
                if st.session_state["page"]
                == "Dashboard"
                else "secondary"
            ),
        ):

            _chuyen_trang(
                "Dashboard"
            )

            st.rerun()

        # ====================================================
        # PHÂN TÍCH CỔ PHIẾU
        # ====================================================

        if st.button(
            "📈 Phân tích cổ phiếu",
            key="nut_phan_tich",
            width="stretch",
            type=(
                "primary"
                if st.session_state["page"]
                == "Phân tích cổ phiếu"
                else "secondary"
            ),
        ):

            _chuyen_trang(
                "Phân tích cổ phiếu"
            )

            st.rerun()

        # ====================================================
        # TIN TỨC
        # ====================================================

        if st.button(
            "📰 Tin tức thị trường",
            key="nut_tin_tuc",
            width="stretch",
            type=(
                "primary"
                if st.session_state["page"]
                == "Tin tức thị trường"
                else "secondary"
            ),
        ):

            _chuyen_trang(
                "Tin tức thị trường"
            )

            st.rerun()

        # ====================================================
        # DANH MỤC
        # ====================================================

        if st.button(
            "💼 Danh mục",
            key="nut_danh_muc",
            width="stretch",
            type=(
                "primary"
                if st.session_state["page"]
                == "Danh mục"
                else "secondary"
            ),
        ):

            _chuyen_trang(
                "Danh mục"
            )

            st.rerun()

        # ====================================================
        # AI
        # ====================================================

        if st.button(
            "🤖 AI Assistant",
            key="nut_ai",
            width="stretch",
            type=(
                "primary"
                if st.session_state["page"]
                == "AI Assistant"
                else "secondary"
            ),
        ):

            _chuyen_trang(
                "AI Assistant"
            )

            st.rerun()

        # ====================================================
        # TRẠNG THÁI
        # ====================================================

        st.markdown(
            "---"
        )

        st.markdown(
            """
            <div style="
                padding: 10px 4px;
                color: #64748b;
                font-size: 11px;
                line-height: 1.7;
            ">
                <b style="color:#cbd5e1;">
                    BRIAN STOCK
                </b>
                <br>
                Dữ liệu thị trường thật
                <br>
                Phân tích kỹ thuật
                <br>
                Định lượng
                <br>
                AI hỗ trợ nghiên cứu
            </div>
            """,
            unsafe_allow_html=True,
        )
