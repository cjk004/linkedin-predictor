# Final Project Part 2: Who's a LinkedIn user?
# Chris Kim
#
# run locally with:  streamlit run app.py
# social_media_usage.csv has to sit in the same folder

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

st.set_page_config(page_title="Who's a LinkedIn user?", page_icon="💼", layout="wide")


# ================================================================ constants
FEATURES = ["income", "education", "parent", "married", "female", "age"]

# the number is the survey code the model was trained on, the text is what the user sees
INCOME_LABELS = {
    1: "Less than $10,000",
    2: "$10,000 to under $20,000",
    3: "$20,000 to under $30,000",
    4: "$30,000 to under $40,000",
    5: "$40,000 to under $50,000",
    6: "$50,000 to under $75,000",
    7: "$75,000 to under $100,000",
    8: "$100,000 to under $150,000",
    9: "$150,000 or more",
}
EDUCATION_LABELS = {
    1: "Less than high school",
    2: "High school incomplete",
    3: "High school graduate",
    4: "Some college, no degree",
    5: "Two-year associate degree",
    6: "Four-year college degree",
    7: "Some postgraduate schooling, no degree",
    8: "Postgraduate or professional degree",
}

# the other platforms in the survey, for the "what else does this segment use" chart
PLATFORMS = {
    "web1a": "Twitter", "web1b": "Instagram", "web1c": "Facebook", "web1d": "Snapchat",
    "web1e": "YouTube", "web1f": "WhatsApp", "web1g": "Pinterest", "web1h": "LinkedIn",
    "web1i": "Reddit", "web1j": "TikTok", "web1k": "Nextdoor",
}

# segment buckets for the survey breakdown tab
AGE_EDGES, AGE_GROUPS = [17, 29, 44, 64, 200], ["18-29", "30-44", "45-64", "65+"]
INCOME_EDGES, INCOME_TIERS = [0, 3, 6, 9], ["Under $30k", "$30k-$75k", "$75k+"]
EDU_EDGES, EDU_TIERS = [0, 3, 5, 6, 8], ["HS or less", "Some college", "Bachelor's", "Grad school"]

# a few ready-made profiles so the demo doesn't start from a blank form
PRESETS = {
    "Example from the assignment": dict(income=8, education=7, age=42, gender="Female", married="Yes", parent="No"),
    "Recent grad":                  dict(income=4, education=6, age=24, gender="Male",   married="No",  parent="No"),
    "Mid-career parent":            dict(income=7, education=6, age=38, gender="Female", married="Yes", parent="Yes"),
    "Retiree":                      dict(income=6, education=5, age=70, gender="Male",   married="Yes", parent="No"),
}

# one accent color for the thing that matters, gray for context, red only for "pushes it down"
BLUE = "#2a78d6"
LIGHT_BLUE = "#cde2fb"
GRAY = "#c3c2b7"
INK = "#0b0b0b"
MUTED = "#898781"
RED = "#e34948"


# ================================================================ data and model
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x


@st.cache_data
def load_data():
    # same cleaning as the part 1 notebook. don't know / refused become NaN and get dropped
    s = pd.read_csv("social_media_usage.csv")

    ss = pd.DataFrame({
        "sm_li":     np.where(s["web1h"] > 2, np.nan, clean_sm(s["web1h"])),
        "income":    np.where(s["income"] > 9, np.nan, s["income"]),
        "education": np.where(s["educ2"] > 8, np.nan, s["educ2"]),
        "parent":    np.where(s["par"] > 2, np.nan, clean_sm(s["par"])),
        "married":   np.where(s["marital"] > 6, np.nan, clean_sm(s["marital"])),
        "female":    np.where(s["gender"] > 3, np.nan, np.where(s["gender"] == 2, 1, 0)),
        "age":       np.where(s["age"] > 97, np.nan, s["age"]),
    })
    keep = ss.notna().all(axis=1)
    ss = ss[keep].astype(int)

    # segment labels for the survey breakdown tab
    ss["age_group"] = pd.cut(ss["age"], AGE_EDGES, labels=AGE_GROUPS).astype(str)
    ss["income_tier"] = pd.cut(ss["income"], INCOME_EDGES, labels=INCOME_TIERS).astype(str)
    ss["education_tier"] = pd.cut(ss["education"], EDU_EDGES, labels=EDU_TIERS).astype(str)

    # yes/no for every platform, same people. 1 = uses it, 0 = doesn't, NaN = didn't answer
    platforms = s.loc[keep, list(PLATFORMS)].rename(columns=PLATFORMS)
    platforms = platforms.where(platforms <= 2)
    platforms = (platforms == 1).astype(float).where(platforms.notna())

    return ss, platforms


@st.cache_resource
def fit_model():
    ss, _ = load_data()
    y = ss["sm_li"]
    X = ss[FEATURES]

    # same split and seed as part 1, so the app and the notebook report the same numbers
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    lr = LogisticRegression(class_weight="balanced")
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    scores = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "cm": confusion_matrix(y_test, y_pred),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }
    return lr, scores


ss, platforms = load_data()
lr, scores = fit_model()
base_rate = ss["sm_li"].mean()


# ================================================================ small helpers
def show_chart(chart):
    # newer streamlit wants width="stretch", older versions only know use_container_width. handle both
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


def bucket(value, edges, labels):
    # which segment a single value lands in
    return str(pd.cut([value], edges, labels=labels)[0])


# ================================================================ sidebar: describe the person
# every control has a key so the preset picker can set it. defaults go in once, on the first run
defaults = dict(preset="Example from the assignment", **PRESETS["Example from the assignment"])
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def apply_preset():
    if st.session_state.preset in PRESETS:
        for k, v in PRESETS[st.session_state.preset].items():
            st.session_state[k] = v


def mark_custom():
    # the user moved a control, so the preset box shouldn't claim it's still that preset
    st.session_state.preset = "Custom"


st.sidebar.header("Describe the person")

st.sidebar.selectbox("Start from a preset", list(PRESETS) + ["Custom"], key="preset", on_change=apply_preset)

st.sidebar.selectbox("Household income", list(INCOME_LABELS), format_func=lambda k: INCOME_LABELS[k],
                     key="income", on_change=mark_custom)
st.sidebar.selectbox("Education", list(EDUCATION_LABELS), format_func=lambda k: EDUCATION_LABELS[k],
                     key="education", on_change=mark_custom)
st.sidebar.slider("Age", 18, 97, key="age", on_change=mark_custom)
st.sidebar.radio("Gender", ["Female", "Male", "Other"], key="gender", horizontal=True, on_change=mark_custom)
st.sidebar.radio("Married?", ["Yes", "No"], key="married", horizontal=True, on_change=mark_custom)
st.sidebar.radio("Parent of a child under 18 at home?", ["Yes", "No"], key="parent", horizontal=True,
                 on_change=mark_custom)

st.sidebar.caption("The model only knows female vs. not, married vs. not, and parent vs. not, "
                   "so \"Male\" and \"Other\" are the same to it.")

# turn the choices into the codes the model expects
income = st.session_state.income
education = st.session_state.education
age = st.session_state.age
female = 1 if st.session_state.gender == "Female" else 0
married = 1 if st.session_state.married == "Yes" else 0
parent = 1 if st.session_state.parent == "Yes" else 0

person = pd.DataFrame([{"income": income, "education": education, "parent": parent,
                        "married": married, "female": female, "age": age}])

prob = lr.predict_proba(person[FEATURES])[0, 1]
pred = int(lr.predict(person[FEATURES])[0])


def prob_for(**changes):
    # same person with one or two things changed
    row = person.copy()
    for k, v in changes.items():
        row[k] = v
    return lr.predict_proba(row[FEATURES])[0, 1]


# ================================================================ headline result
st.title("Who's a LinkedIn user?")
st.write(
    "Describe a customer on the left. The model says whether that person is likely to be on LinkedIn "
    "and how likely. The tabs below show what drives that answer, where the person's segment sits in the "
    "survey, and how good the model is."
)

c1, c2, c3 = st.columns([1.1, 1.3, 0.9], gap="large")

with c1:
    if pred == 1:
        st.success("### Likely a LinkedIn user")
    else:
        st.error("### Probably not a LinkedIn user")
    st.caption("The model calls anyone above 50% a user.")

with c2:
    st.metric("Probability this person uses LinkedIn", f"{prob:.0%}",
              delta=f"{prob - base_rate:+.0%} vs. survey average")

    # a meter: filled to the probability, with the survey average and the 50% cutoff ticked on the track
    meter = pd.DataFrame({"lo": [0.0], "hi": [1.0], "p": [prob]})
    marks = pd.DataFrame({"x": [base_rate, 0.5], "label": [f"survey avg {base_rate:.0%}", "cutoff 50%"],
                          "align": ["right", "left"], "dx": [-5, 5]})
    track = alt.Chart(meter).mark_bar(color=LIGHT_BLUE, cornerRadius=4).encode(
        x=alt.X("lo:Q", scale=alt.Scale(domain=[0, 1]), axis=None), x2="hi:Q",
        y=alt.value(6), y2=alt.value(26))
    fill = alt.Chart(meter).mark_bar(color=BLUE, cornerRadius=4).encode(
        x="lo:Q", x2="p:Q", y=alt.value(6), y2=alt.value(26),
        tooltip=[alt.Tooltip("p:Q", title="probability", format=".1%")])
    ticks = alt.Chart(marks).mark_rule(color=INK, strokeWidth=2).encode(
        x="x:Q", y=alt.value(2), y2=alt.value(30))
    tick_labels = alt.Chart(marks).mark_text(color=MUTED, fontSize=11, baseline="middle",
                                             align=alt.expr("datum.align"), dx=alt.expr("datum.dx")).encode(
        x="x:Q", y=alt.value(42), text="label:N")
    show_chart((track + fill + ticks + tick_labels).properties(height=54).configure_view(strokeWidth=0))

with c3:
    st.metric("Survey average", f"{base_rate:.0%}")
    st.caption(f"{int(ss['sm_li'].sum())} of the {len(ss):,} people in the cleaned survey use LinkedIn.")


# ================================================================ tabs
tab_person, tab_segments, tab_model = st.tabs(["This person", "Segments and platforms", "About the model"])


# ---------------------------------------------------------------- tab 1: this person
with tab_person:
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Same person at every age")

        ages = pd.DataFrame({"income": income, "education": education, "parent": parent,
                             "married": married, "female": female, "age": range(18, 98)})
        ages["prob"] = lr.predict_proba(ages[FEATURES])[:, 1]

        # age pushes the probability down, so the crossover is the first age under 50%
        below = ages.loc[ages["prob"] < 0.5, "age"]
        if below.empty:
            crossing = "This profile stays above 50% at every age in the survey."
        elif below.min() == 18:
            crossing = "This profile is under 50% at every age in the survey."
        else:
            crossing = f"For this profile the model flips to \"not a user\" at age {int(below.min())}."
        st.caption(crossing + " Everything except age is held at the settings on the left.")

        # label each reference line on whichever end is farther from the curve, so the text stays off the line
        refs = pd.DataFrame({"y": [0.5, base_rate], "label": ["50% cutoff", f"survey average {base_rate:.0%}"]})
        p_first, p_last = ages["prob"].iloc[0], ages["prob"].iloc[-1]
        refs["left"] = (refs["y"] - p_first).abs() > (refs["y"] - p_last).abs()
        refs["x"] = np.where(refs["left"], 18, 97)
        refs["align"] = np.where(refs["left"], "left", "right")
        refs["dx"] = np.where(refs["left"], 4, -4)

        line = alt.Chart(ages).mark_line(color=BLUE, strokeWidth=2.5).encode(
            x=alt.X("age:Q", title="Age", scale=alt.Scale(domain=[18, 97], nice=False)),
            y=alt.Y("prob:Q", title="Probability of using LinkedIn", axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, 1])))
        hover = alt.Chart(ages).mark_circle(size=250, opacity=0).encode(
            x="age:Q", y="prob:Q",
            tooltip=[alt.Tooltip("age:Q", title="age"), alt.Tooltip("prob:Q", title="probability", format=".1%")])
        ref_lines = alt.Chart(refs).mark_rule(color=MUTED, strokeDash=[5, 4]).encode(y="y:Q")
        ref_text = alt.Chart(refs).mark_text(color=MUTED, fontSize=11, dy=-7, align=alt.expr("datum.align"),
                                             dx=alt.expr("datum.dx")).encode(x="x:Q", y="y:Q", text="label:N")
        dot = alt.Chart(pd.DataFrame({"age": [age], "prob": [prob]})).mark_circle(
            size=170, color=BLUE, stroke="white", strokeWidth=2).encode(
            x="age:Q", y="prob:Q",
            tooltip=[alt.Tooltip("age:Q", title="age"), alt.Tooltip("prob:Q", title="probability", format=".1%")])

        show_chart((line + hover + ref_lines + ref_text + dot).properties(height=320))

    with right:
        st.subheader("Change one thing")
        st.caption("This same person with one detail changed. Blue raises the probability, red lowers it.")

        rows = []
        if income < 9:
            rows.append(("One income bracket up", INCOME_LABELS[income + 1], prob_for(income=income + 1)))
        else:
            rows.append(("One income bracket down", INCOME_LABELS[income - 1], prob_for(income=income - 1)))
        if education < 8:
            rows.append(("One education level up", EDUCATION_LABELS[education + 1],
                         prob_for(education=education + 1)))
        else:
            rows.append(("One education level down", EDUCATION_LABELS[education - 1],
                         prob_for(education=education - 1)))
        if age - 10 >= 18:
            rows.append(("10 years younger", f"age {age - 10}", prob_for(age=age - 10)))
        if age + 10 <= 97:
            rows.append(("10 years older", f"age {age + 10}", prob_for(age=age + 10)))
        rows.append(("Not a parent" if parent else "Parent", "", prob_for(parent=1 - parent)))
        rows.append(("Not married" if married else "Married", "", prob_for(married=1 - married)))
        rows.append(("Male or other" if female else "Female", "", prob_for(female=1 - female)))

        whatif = pd.DataFrame(rows, columns=["scenario", "detail", "new_prob"])
        whatif["change"] = whatif["new_prob"] - prob
        whatif["direction"] = np.where(whatif["change"] >= 0, "raises it", "lowers it")
        whatif["label"] = whatif["change"].map(lambda c: f"{c * 100:+.1f} pts")
        # sort biggest gain to biggest loss, and leave room past the bar ends for the labels
        by_change = whatif.sort_values("change", ascending=False)["scenario"].tolist()
        reach = whatif["change"].abs().max() * 1.45

        bars = alt.Chart(whatif).mark_bar(cornerRadiusEnd=4, size=18).encode(
            x=alt.X("change:Q", title="Change in probability", axis=alt.Axis(format="+%", tickCount=5),
                    scale=alt.Scale(domain=[-reach, reach])),
            y=alt.Y("scenario:N", title=None, sort=by_change, axis=alt.Axis(labelLimit=200)),
            color=alt.Color("direction:N", legend=None,
                            scale=alt.Scale(domain=["raises it", "lowers it"], range=[BLUE, RED])),
            tooltip=[alt.Tooltip("scenario:N", title="change"), alt.Tooltip("detail:N", title="to"),
                     alt.Tooltip("new_prob:Q", title="new probability", format=".1%"),
                     alt.Tooltip("change:Q", title="change", format="+.1%")])
        labels = alt.Chart(whatif).mark_text(color=INK, fontSize=11,
                                             dx=alt.expr("datum.change >= 0 ? 6 : -6"),
                                             align=alt.expr("datum.change >= 0 ? 'left' : 'right'")).encode(
            x="change:Q", y=alt.Y("scenario:N", sort=by_change), text="label:N")
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=MUTED).encode(x="x:Q")

        show_chart((bars + labels + zero).properties(height=320))


# ---------------------------------------------------------------- tab 2: segments and platforms
with tab_segments:
    st.write("The model scores one person at a time. This tab steps back to the survey itself: where the LinkedIn "
             "users are concentrated, and what else the people in this person's segment use. "
             "These are survey shares, not model predictions.")

    breakdown = st.radio("Break the survey down by", ["Income", "Education"], horizontal=True)

    if breakdown == "Income":
        col, order, my_tier = "income_tier", INCOME_TIERS, bucket(income, INCOME_EDGES, INCOME_TIERS)
        label_expr = "datum.label"
    else:
        col, order, my_tier = "education_tier", EDU_TIERS, bucket(education, EDU_EDGES, EDU_TIERS)
        # four columns is a squeeze, so the education labels wrap onto two lines
        label_expr = ("datum.label == 'HS or less' ? ['HS or', 'less'] : "
                      "datum.label == 'Some college' ? ['Some', 'college'] : "
                      "datum.label == 'Grad school' ? ['Grad', 'school'] : datum.label")
    my_age_group = bucket(age, AGE_EDGES, AGE_GROUPS)

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Where the LinkedIn users are")
        st.caption("Share of each segment that uses LinkedIn, with the number of survey respondents underneath. "
                   "The outlined cell is the person on the left.")

        cells = ss.groupby(["age_group", col], as_index=False)["sm_li"].agg(share="mean", people="count")
        cells["selected"] = (cells["age_group"] == my_age_group) & (cells[col] == my_tier)
        cells["label"] = cells["share"].map(lambda v: f"{v:.0%}")
        cells["n_label"] = cells["people"].map(lambda n: f"n = {n}")
        grid = dict(x=alt.X(f"{col}:N", title=breakdown, sort=order,
                            axis=alt.Axis(labelAngle=0, labelOverlap=False, labelExpr=label_expr)),
                    y=alt.Y("age_group:N", title="Age", sort=AGE_GROUPS))
        dark_cell = alt.datum.share > 0.45

        heat = alt.Chart(cells).mark_rect(stroke="white", strokeWidth=2).encode(
            color=alt.Color("share:Q", scale=alt.Scale(scheme="blues", domain=[0, 0.7]), legend=None),
            tooltip=[alt.Tooltip("age_group:N", title="age"), alt.Tooltip(f"{col}:N", title=breakdown.lower()),
                     alt.Tooltip("share:Q", title="use LinkedIn", format=".1%"),
                     alt.Tooltip("people:Q", title="people in survey")],
            **grid)
        share_text = alt.Chart(cells).mark_text(fontSize=14, dy=-7).encode(
            text="label:N", color=alt.condition(dark_cell, alt.value("white"), alt.value(INK)), **grid)
        n_text = alt.Chart(cells).mark_text(fontSize=10, dy=11, opacity=0.85).encode(
            text="n_label:N", color=alt.condition(dark_cell, alt.value("white"), alt.value(INK)), **grid)
        outline = alt.Chart(cells).mark_rect(fillOpacity=0, stroke=INK, strokeWidth=3).encode(
            **grid).transform_filter("datum.selected")

        show_chart((heat + share_text + n_text + outline).properties(height=300))

    with right:
        seg = ss[(ss["age_group"] == my_age_group) & (ss[col] == my_tier)]
        seg_share = seg["sm_li"].mean()

        st.subheader(f"What this segment uses: {my_age_group}, {my_tier.lower()}")
        st.caption(f"{len(seg)} survey respondents in this segment. Bars are the segment, the dark tick is "
                   f"all {len(ss):,} respondents. Hover a bar for both numbers.")

        plat = pd.DataFrame({"segment": platforms.loc[seg.index].mean(), "everyone": platforms.mean()})
        plat = plat.reset_index().rename(columns={"index": "platform"})
        plat["series"] = np.where(plat["platform"] == "LinkedIn", "LinkedIn", "Other platforms")
        plat["label"] = plat["segment"].map(lambda v: f"{v:.0%}")
        plat_order = plat.sort_values("segment", ascending=False)["platform"].tolist()
        everyone_order = plat.sort_values("everyone", ascending=False)["platform"].tolist()

        # bars are the segment, ticks are the whole survey. one shared color field so the legend covers both
        long = pd.concat([
            plat.assign(value=plat["segment"]),
            plat.assign(value=plat["everyone"], series="All respondents"),
        ])
        palette = alt.Scale(domain=["LinkedIn", "Other platforms", "All respondents"], range=[BLUE, GRAY, INK])
        legend = alt.Legend(title=None, orient="bottom", columns=2)

        seg_bars = alt.Chart(long).transform_filter(alt.datum.series != "All respondents").mark_bar(
            cornerRadiusEnd=4, size=16).encode(
            x=alt.X("value:Q", title="Share who use it", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("platform:N", title=None, sort=plat_order, axis=alt.Axis(labelOverlap=False)),
            color=alt.Color("series:N", scale=palette, legend=legend),
            tooltip=[alt.Tooltip("platform:N"), alt.Tooltip("segment:Q", title="this segment", format=".1%"),
                     alt.Tooltip("everyone:Q", title="everyone", format=".1%")])
        all_ticks = alt.Chart(long).transform_filter(alt.datum.series == "All respondents").mark_tick(
            thickness=2, size=22).encode(
            x="value:Q", y=alt.Y("platform:N", sort=plat_order),
            color=alt.Color("series:N", scale=palette, legend=legend),
            tooltip=[alt.Tooltip("platform:N"), alt.Tooltip("segment:Q", title="this segment", format=".1%"),
                     alt.Tooltip("everyone:Q", title="everyone", format=".1%")])
        li_label = alt.Chart(plat[plat["platform"] == "LinkedIn"]).mark_text(
            color=INK, fontSize=11, align="left", dx=6).encode(
            x="segment:Q", y=alt.Y("platform:N", sort=plat_order), text="label:N")

        show_chart((seg_bars + all_ticks + li_label).properties(height=380))

        st.caption(f"{seg_share:.0%} of this segment uses LinkedIn (the model gives this exact person {prob:.0%}). "
                   f"LinkedIn ranks #{plat_order.index('LinkedIn') + 1} of 11 platforms in the segment, "
                   f"#{everyone_order.index('LinkedIn') + 1} in the survey overall.")


# ---------------------------------------------------------------- tab 3: about the model
with tab_model:
    cm = scores["cm"]
    tn, fp, fn, tp = cm.ravel()

    st.write(f"Logistic regression on six features: income (1 to 9), education (1 to 8), parent, married, female, "
             f"and age. Trained on {scores['n_train']} people, scored on {scores['n_test']} it never saw. "
             f"`class_weight='balanced'` so the third who use LinkedIn count as much as the two thirds who don't, "
             f"otherwise the model just says \"no\" to almost everyone.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{scores['accuracy']:.1%}", help="Share of the test set the model got right")
    m2.metric("Recall", f"{scores['recall']:.1%}", help="Of the real LinkedIn users, how many it found")
    m3.metric("Precision", f"{scores['precision']:.1%}", help="Of the people it called users, how many really are")
    m4.metric("Base rate", f"{base_rate:.1%}", help="Share of everyone who uses LinkedIn. Picking at random gets this")

    st.caption(f"Accuracy is about what you'd get by calling everyone a non-user ({1 - base_rate:.0%}), but that "
               f"model finds nobody. This one finds {tp} of the {tp + fn} real users in the test set, and about "
               f"half the people it flags really are users, against a third at random. For deciding where to put "
               f"ad money, finding the users matters more than a couple of points of accuracy.")

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("What moves the prediction")
        st.caption("How much the odds of being a LinkedIn user change for one step up in each feature. "
                   "Blue pushes toward LinkedIn, red pushes away.")

        nice = {"income": "Income (per bracket)", "education": "Education (per level)", "age": "Age (per year)",
                "parent": "Parent", "married": "Married", "female": "Female"}
        odds = pd.DataFrame({"feature": [nice[f] for f in FEATURES], "odds_ratio": np.exp(lr.coef_[0])})
        odds["pct"] = odds["odds_ratio"] - 1
        odds["direction"] = np.where(odds["pct"] >= 0, "more likely", "less likely")
        odds["label"] = odds["pct"].map(lambda v: f"{v:+.0%}")
        by_pct = odds.sort_values("pct", ascending=False)["feature"].tolist()
        reach = odds["pct"].abs().max() * 1.4

        obars = alt.Chart(odds).mark_bar(cornerRadiusEnd=4, size=18).encode(
            x=alt.X("pct:Q", title="Change in odds per one step up", axis=alt.Axis(format="+%"),
                    scale=alt.Scale(domain=[-reach, reach])),
            y=alt.Y("feature:N", title=None, sort=by_pct, axis=alt.Axis(labelLimit=200)),
            color=alt.Color("direction:N", legend=None,
                            scale=alt.Scale(domain=["more likely", "less likely"], range=[BLUE, RED])),
            tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("odds_ratio:Q", title="odds ratio", format=".2f"),
                     alt.Tooltip("pct:Q", title="change in odds", format="+.0%")])
        olabels = alt.Chart(odds).mark_text(color=INK, fontSize=11,
                                            dx=alt.expr("datum.pct >= 0 ? 6 : -6"),
                                            align=alt.expr("datum.pct >= 0 ? 'left' : 'right'")).encode(
            x="pct:Q", y=alt.Y("feature:N", sort=by_pct), text="label:N")
        ozero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=MUTED).encode(x="x:Q")

        show_chart((obars + olabels + ozero).properties(height=250))
        st.caption("Age looks small because it's per year - forty years of about -3% each compounds into the drop "
                   "on the first tab. Married comes out negative even though married people use LinkedIn more in "
                   "the raw data: they're older and higher income, and the model gives the credit to those.")

    with right:
        st.subheader("Test set confusion matrix")
        st.caption(f"Rows are what people actually said, columns are what the model predicted. "
                   f"{tn + tp} of {scores['n_test']} are on the diagonal.")

        cm_long = pd.DataFrame({
            "actual": ["Not a user", "Not a user", "LinkedIn user", "LinkedIn user"],
            "predicted": ["Not a user", "LinkedIn user", "Not a user", "LinkedIn user"],
            "count": [tn, fp, fn, tp],
            "what": ["true negative", "false positive", "false negative", "true positive"],
        })
        cm_grid = dict(x=alt.X("predicted:N", title="Predicted", sort=["Not a user", "LinkedIn user"],
                               axis=alt.Axis(labelAngle=0, orient="top")),
                       y=alt.Y("actual:N", title="Actual", sort=["Not a user", "LinkedIn user"]))
        dark_cm = alt.datum.count > 80

        cm_heat = alt.Chart(cm_long).mark_rect(stroke="white", strokeWidth=2).encode(
            color=alt.Color("count:Q", legend=None, scale=alt.Scale(scheme="blues")),
            tooltip=[alt.Tooltip("what:N", title="quadrant"), alt.Tooltip("count:Q", title="people")], **cm_grid)
        cm_text = alt.Chart(cm_long).mark_text(fontSize=24, dy=-6).encode(
            text="count:Q", color=alt.condition(dark_cm, alt.value("white"), alt.value(INK)), **cm_grid)
        cm_sub = alt.Chart(cm_long).mark_text(fontSize=11, dy=16, opacity=0.85).encode(
            text="what:N", color=alt.condition(dark_cm, alt.value("white"), alt.value(INK)), **cm_grid)

        show_chart((cm_heat + cm_text + cm_sub).properties(height=250))

    st.caption("Part 1 of this project (the notebook) has the full cleaning, exploratory analysis, and the by-hand "
               "precision, recall and F1 calculations.")
