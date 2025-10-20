from lxml import etree
import os
import copy
import glob
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def parse_xml_with_recovery(xml_file_path):
    parser = etree.XMLParser(recover=True)
    tree = etree.parse(xml_file_path, parser)
    return tree.getroot()

def parse_poster_xml(xml_file):
    """
    Parse an XML describing a single poster layout, e.g.:

    <Poster Width="685" Height="968">
      <Panel left="5" right="160" width="674" height="123">
        <Text>Introduction</Text>
        <Figure left="567" right="178" width="81" height="99" no="1" ... />
      </Panel>
      ...
    </Poster>

    Returns a dict with:
      {
        'poster_width': float,
        'poster_height': float,
        'panels': [
          {
            'x': float,
            'y': float,
            'width': float,
            'height': float,
            'text_blocks': [string, string, ...],
            'figure_blocks': [(fx, fy, fw, fh), ...]
          },
          ...
        ]
      }
    """
    root = parse_xml_with_recovery(xml_file)

    # Poster dimensions
    poster_w = float(root.get("Width", "1"))
    poster_h = float(root.get("Height", "1"))

    panels_data = []

    # Iterate <Panel> elements
    for panel_node in root.findall("Panel"):
        x = float(panel_node.get("left", "0"))
        y = float(panel_node.get("right", "0"))
        w = float(panel_node.get("width", "0"))
        h = float(panel_node.get("height", "0"))

        # Gather text blocks
        text_blocks = []
        for text_node in panel_node.findall("Text"):
            txt = text_node.text or ""
            txt = txt.strip()
            if txt:
                text_blocks.append(txt)

        # Gather figure blocks
        figure_blocks = []
        for fig_node in panel_node.findall("Figure"):
            fx = float(fig_node.get("left", "0"))
            fy = float(fig_node.get("right", "0"))
            fw = float(fig_node.get("width", "0"))
            fh = float(fig_node.get("height", "0"))
            figure_blocks.append((fx, fy, fw, fh))

        panel_info = {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "text_blocks": text_blocks,
            "figure_blocks": figure_blocks
        }
        panels_data.append(panel_info)

    return {
        "poster_width": poster_w,
        "poster_height": poster_h,
        "panels": panels_data
    }

def compute_panel_attributes(poster_data):
    """
    Given poster_data, compute:
      - tp: ratio of text length for each panel
      - gp: ratio of figure area for each panel
      - sp: ratio of panel area to total poster area
      - rp: aspect ratio (width / height)

    Returns a list of dicts, each:
      {
        'tp': float,
        'gp': float,
        'sp': float,
        'rp': float
      }
    """

    poster_w = poster_data["poster_width"]
    poster_h = poster_data["poster_height"]
    panels   = poster_data["panels"]

    poster_area = max(poster_w * poster_h, 1.0)  # avoid zero

    # 1) Compute total text length across all panels
    # 2) Compute total figure area across all panels
    total_text_length  = 0
    total_figure_area  = 0

    # We'll store partial info about each panel so we don't parse multiple times
    panel_list = []
    for p in panels:
        # Combine all text
        panel_text_joined = " ".join(p["text_blocks"])
        panel_text_len = len(panel_text_joined)

        # Sum area of figure blocks
        panel_fig_area = 0.0
        for (fx, fy, fw, fh) in p["figure_blocks"]:
            panel_fig_area += (fw * fh)

        panel_list.append({
            "x": p["x"],
            "y": p["y"],
            "width": p["width"],
            "height": p["height"],
            "text_len": panel_text_len,
            "fig_area": panel_fig_area
        })

        total_text_length  += panel_text_len
        total_figure_area  += panel_fig_area

    # Avoid divide by zero
    if total_text_length < 1:
        total_text_length = 1
    if total_figure_area < 1e-9:
        total_figure_area = 1e-9

    # 3) Compute attributes
    results = []
    for pinfo in panel_list:
        pw = pinfo["width"]
        ph = pinfo["height"]

        panel_area = pw * ph
        sp = panel_area / poster_area  # fraction of total area
        rp = (pw / ph) if ph > 0 else 1.0

        tp = pinfo["text_len"] / float(total_text_length)
        gp = pinfo["fig_area"] / float(total_figure_area)

        results.append({
            "tp": tp,
            "gp": gp,
            "sp": sp,
            "rp": rp
        })

    return results

def train_panel_attribute_inference(panel_records):
    """
    The training data `panel_records` is a list of dicts, each containing:
      {
        'tp': float,
        'gp': float,
        'sp': float,  # (label for the sp regression)
        'rp': float   # (label for the rp regression)
      }

    We'll train two linear regressors:
       sp = w_s * [tp, gp, 1]
       rp = w_r * [tp, gp, 1]

    Returns dict with learned parameters:
      {
        'w_s': array,  # shape (3,) => sp = w_s[0]*tp + w_s[1]*gp + w_s[2]
        'sigma_s': float,  # variance of residual for sp
        'w_r': array,
        'sigma_r': float
      }
    """
    # Build data arrays
    X_list  = []
    sp_list = []
    rp_list = []

    for rec in panel_records:
        tp = rec['tp']
        gp = rec['gp']
        sp = rec['sp']
        rp = rec['rp']
        # X = [tp, gp, 1]
        X_list.append([tp, gp, 1.0])
        sp_list.append(sp)
        rp_list.append(rp)

    X_array = np.array(X_list, dtype=float)
    y_sp = np.array(sp_list, dtype=float)
    y_rp = np.array(rp_list, dtype=float)

    # Fit linear regression for sp
    linreg_sp = LinearRegression(fit_intercept=False)
    linreg_sp.fit(X_array, y_sp)
    w_s = linreg_sp.coef_
    pred_sp = linreg_sp.predict(X_array)
    residual_sp = y_sp - pred_sp
    sigma_s = np.var(residual_sp, ddof=1)

    # Fit linear regression for rp
    linreg_rp = LinearRegression(fit_intercept=False)
    linreg_rp.fit(X_array, y_rp)
    w_r = linreg_rp.coef_
    pred_rp = linreg_rp.predict(X_array)
    residual_rp = y_rp - pred_rp
    sigma_r = np.var(residual_rp, ddof=1)

    model_params = {
        "w_s": w_s,
        "sigma_s": sigma_s,
        "w_r": w_r,
        "sigma_r": sigma_r
    }
    return model_params


def parse_poster_xml_for_figures(xml_path):
    root = parse_xml_with_recovery(xml_path)

    poster_w = float(root.get("Width", "1"))
    poster_h = float(root.get("Height", "1"))
    poster_area = poster_w * poster_h

    records = []

    for panel in root.findall("Panel"):
        px, py = float(panel.get("left", 0)), float(panel.get("right", 0))
        pw, ph = float(panel.get("width", 1)), float(panel.get("height", 1))
        panel_area = pw * ph
        sp = panel_area / poster_area
        rp = pw / ph if ph > 0 else 1.0

        lp = sum(len(t.text.strip()) for t in panel.findall("Text") if t.text)

        for fig in panel.findall("Figure"):
            fx, fy = float(fig.get("left", 0)), float(fig.get("right", 0))
            fw, fh = float(fig.get("width", 1)), float(fig.get("height", 1))

            sg = (fw * fh) / poster_area
            rg = fw / fh if fh > 0 else 1.0
            ug = fw / pw if pw > 0 else 0.1

            panel_center_x = px + pw / 2
            fig_center_x = fx + fw / 2
            delta_x = fig_center_x - panel_center_x

            hg = 0 if delta_x < -pw / 6 else (2 if delta_x > pw / 6 else 1)

            record = {"sp": sp, "rp": rp, "lp": lp, "sg": sg, "rg": rg, "hg": hg, "ug": ug}
            records.append(record)

    return records


def train_figure_model(figure_records):
    X_hg, y_hg, X_ug, y_ug = [], [], [], []
    for r in figure_records:
        feats = [r["sp"], r["lp"], r["sg"], 1.0]
        X_hg.append(feats)
        y_hg.append(r["hg"])
        X_ug.append(feats)
        y_ug.append(r["ug"])

    clf_hg = LogisticRegression(multi_class="multinomial", solver="lbfgs", fit_intercept=False)
    clf_hg.fit(X_hg, y_hg)

    lin_ug = LinearRegression(fit_intercept=False)
    lin_ug.fit(X_ug, y_ug)
    residuals = y_ug - lin_ug.predict(X_ug)
    sigma_u = np.var(residuals, ddof=1)

    return {
        "clf_hg": clf_hg,
        "w_u": lin_ug.coef_,
        "sigma_u": sigma_u
    }


def main_train():
    poster_dataset_path = 'assets/poster_data/Train'
    # loop through all folders in the dataset
    xml_files = []
    for folder in os.listdir(poster_dataset_path):
        folder_path = os.path.join(poster_dataset_path, folder)
        if os.path.isdir(folder_path):
            # find all XML files in this folder
            xml_files.extend(glob.glob(os.path.join(folder_path, "*.txt")))

    all_panel_records = []
    for xml_file in xml_files:
        poster_data = parse_poster_xml(xml_file)
        # compute tp, gp, sp, rp
        panel_attrs = compute_panel_attributes(poster_data)
        # each panel_attrs entry is {tp, gp, sp, rp}
        all_panel_records.extend(panel_attrs)

    all_figure_records = []
    for xml_path in xml_files:
        recs = parse_poster_xml_for_figures(xml_path)
        all_figure_records.extend(recs)

    panel_model_params = train_panel_attribute_inference(all_panel_records)
    figure_model_params = train_figure_model(all_figure_records)

    return panel_model_params, figure_model_params

def place_text_and_figures_exact(panel_dict, figure_model_params, section_title_height=32):
    """
    Lay out text and figure boxes inside a panel.

    The figure’s aspect ratio (width / height) is now enforced strictly:
        • width  ≤ panel width
        • height ≤ 0.60 × panel height      (empirical upper‑bound you already used)
        • width / height == panel_dict["figure_aspect"]
    """
    # ---------------- Constants used for text layout -----------------
    char_width_px  = 7
    line_height_px = 16
    chars_per_line = max(int(panel_dict["width"] / char_width_px), 1)

    total_lines_text  = np.ceil(panel_dict["text_len"] / chars_per_line)
    total_text_height = total_lines_text * line_height_px

    x_p, y_p = panel_dict["x"], panel_dict["y"]
    w_p, h_p = panel_dict["width"], panel_dict["height"]

    figure_boxes, text_boxes = [], []

    panel_name_lower = panel_dict["panel_name"].lower()
    has_title_in_name = "title" in panel_name_lower

    # -------------------------------------------------------
    # Helper to build a text‑box dict
    # -------------------------------------------------------
    def make_text_box(panel_id, x, y, width, height, textbox_id, textbox_name):
        return {
            "panel_id":   panel_id,
            "x":          float(x),
            "y":          float(y),
            "width":      float(width),
            "height":     float(height),
            "textbox_id": textbox_id,
            "textbox_name": textbox_name,
        }

    # -----------------------------------------------------------------------
    # Case 1 — no figure in this panel
    # -----------------------------------------------------------------------
    if panel_dict["figure_size"] <= 0:
        if has_title_in_name:
            text_boxes.append(
                make_text_box(panel_dict["panel_id"], x_p, y_p, w_p, h_p,
                              textbox_id=0,
                              textbox_name=f'p<{panel_dict["panel_name"]}>_t0')
            )
        else:
            title_h = min(section_title_height, h_p)
            text_boxes.extend([
                make_text_box(panel_dict["panel_id"], x_p, y_p, w_p, title_h,
                              textbox_id=0,
                              textbox_name=f'p<{panel_dict["panel_name"]}>_t0'),
                make_text_box(panel_dict["panel_id"], x_p, y_p + title_h, w_p, h_p - title_h,
                              textbox_id=1,
                              textbox_name=f'p<{panel_dict["panel_name"]}>_t1'),
            ])
        return text_boxes, figure_boxes   # early‑return (simpler branch)

    # -----------------------------------------------------------------------
    # Case 2 — there *is* a figure
    # -----------------------------------------------------------------------
    # 1.  Sample horizontal‑alignment class (hg) and raw width fraction (ug)
    feat      = np.array([panel_dict["sp"],
                          panel_dict["text_len"],
                          panel_dict["figure_size"],
                          1.0]).reshape(1, -1)

    clf_hg    = figure_model_params["clf_hg"]
    hg_sample = int(np.argmax(clf_hg.predict_proba(feat)[0]))

    mean_ug   = float(np.dot(figure_model_params["w_u"], feat.flatten()))
    sigma_u   = float(np.sqrt(figure_model_params["sigma_u"]))
    ug_sample = float(np.clip(np.random.normal(mean_ug, sigma_u), 0.10, 0.80))  # 10‑80 % of width

    # 2.  **Size the figure while *preserving* aspect ratio**
    aspect     = float(panel_dict["figure_aspect"])       # width / height
    fig_w      = ug_sample * w_p                          # preliminary width
    fig_h      = fig_w / aspect

    max_fig_h  = 0.60 * h_p                               # same limit you had
    if fig_h > max_fig_h:                                 # too tall → scale down
        scale  = max_fig_h / fig_h
        fig_w *= scale
        fig_h  = max_fig_h        # (ratio still intact)

    # 3.  Horizontal placement
    if hg_sample == 0:          # left
        fig_x = x_p
    elif hg_sample == 2:        # right
        fig_x = x_p + w_p - fig_w
    else:                       # center
        fig_x = x_p + 0.5 * (w_p - fig_w)
    # Vertical centering
    fig_y = y_p + 0.5 * (h_p - fig_h)

    # 4.  Split text into “top” and “bottom” areas around the figure
    top_text_h    = (fig_y - y_p)
    bottom_text_h = (y_p + h_p) - (fig_y + fig_h)

    # --- build top‑text boxes
    if has_title_in_name:
        text_boxes.append(
            make_text_box(panel_dict["panel_id"], x_p, y_p, w_p, top_text_h,
                          textbox_id=0,
                          textbox_name=f'p<{panel_dict["panel_name"]}>_t0')
        )
        next_id = 1
    else:
        title_h = min(section_title_height, top_text_h)
        text_boxes.extend([
            make_text_box(panel_dict["panel_id"], x_p, y_p, w_p, title_h,
                          textbox_id=0,
                          textbox_name=f'p<{panel_dict["panel_name"]}>_t0'),
            make_text_box(panel_dict["panel_id"], x_p, y_p + title_h, w_p, top_text_h - title_h,
                          textbox_id=1,
                          textbox_name=f'p<{panel_dict["panel_name"]}>_t1'),
        ])
        next_id = 2

    # --- bottom text box
    text_boxes.append(
        make_text_box(panel_dict["panel_id"], x_p, fig_y + fig_h, w_p, bottom_text_h,
                      textbox_id=next_id,
                      textbox_name=f'p<{panel_dict["panel_name"]}>_t{next_id}')
    )

    # 5.  Figure box
    figure_boxes.append({
        "panel_id":   panel_dict["panel_id"],
        "x":          float(fig_x),
        "y":          float(fig_y),
        "width":      float(fig_w),
        "height":     float(fig_h),
        "figure_id":  0,
        "figure_name": f'p<{panel_dict["panel_name"]}>_f0',
    })

    return text_boxes, figure_boxes


def to_inches(value_in_units, units_per_inch=72):
    """
    Convert a single coordinate or dimension from 'units' to inches.
    For example, if your units are 'points' (72 points = 1 inch),
    then units_per_inch=72.
    If your units are 'pixels' at 96 DPI, then units_per_inch=96.
    """
    return value_in_units / units_per_inch


def from_inches(value_in_inches, units_per_inch=72):
    """
    Convert from inches back to the original 'units'.
    """
    return value_in_inches * units_per_inch


def softmax(logits):
    s = sum(np.exp(logits))
    return [np.exp(l)/s for l in logits]


def infer_panel_attrs(panel_model, tp, gp):
    # sp = w_s dot [tp, gp, 1]
    # rp = w_r dot [tp, gp, 1]
    vec = np.array([tp, gp, 1.0])
    w_s = panel_model["w_s"]
    w_r = panel_model["w_r"]
    sp = np.dot(w_s, vec)
    rp = np.dot(w_r, vec)
    # clamp
    sp = max(sp, 0.01)
    rp = max(rp, 0.05)
    return sp, rp


def panel_layout_generation(panels, x, y, w, h):
    # If only 1 panel, place it entirely
    if len(panels) == 1:
        p = panels[0]
        cur_rp = (w/h) if h>1e-9 else p["rp"]
        loss = abs(p["rp"] - cur_rp)
        arrangement = [{
            "panel_name": p["section_name"],
            "panel_id": p["panel_id"],
            "x": x, "y": y,
            "width": w, "height": h
        }]
        return loss, arrangement

    best_loss = float('inf')
    best_arr = []
    total_sp = sum(pp["sp"] for pp in panels)
    n = len(panels)

    for i in range(1, n):
        subset1 = panels[:i]
        subset2 = panels[i:]
        sp1 = sum(pp["sp"] for pp in subset1)
        ratio = sp1 / total_sp

        # horizontal
        h_top = ratio * h
        if 0 < h_top < h:
            l1, a1 = panel_layout_generation(subset1, x, y, w, h_top)
            l2, a2 = panel_layout_generation(subset2, x, y + h_top, w, h - h_top)
            if (l1 + l2) < best_loss:
                best_loss = l1 + l2
                best_arr = a1 + a2

        # vertical
        w_left = ratio * w
        if 0 < w_left < w:
            l1, a1 = panel_layout_generation(subset1, x, y, w_left, h)
            l2, a2 = panel_layout_generation(subset2, x + w_left, y, w - w_left, h)
            if (l1 + l2) < best_loss:
                best_loss = l1 + l2
                best_arr = a1 + a2

    return best_loss, best_arr

def split_textbox(textbox, ratio):
    """
    Splits a textbox dictionary horizontally into two parts.
    
    Parameters:
      textbox (dict): A dictionary with the keys
                      'panel_id', 'x', 'y', 'width', 'height', 'textbox_id', 'textbox_name'
      ratio (float or int): Ratio of top height to bottom height.
                            For example, if ratio is 3, then:
                              top_height = (3/4) * height
                              bottom_height = (1/4) * height
                              
    Returns:
      tuple: Two dictionaries corresponding to the top and bottom split textboxes.
    """
    # Calculate the new heights
    total_ratio = ratio + 1  # because the ratio represents top:bottom as (ratio):(1)
    top_height = textbox['height'] * ratio / total_ratio
    bottom_height = textbox['height'] * 1 / total_ratio

    # Derive the base textbox name by splitting off the existing _t suffix if present.
    # This assumes the original textbox_name ends with "_t<number>".
    base_name = textbox['textbox_name'].rsplit('_t', 1)[0]

    # Create the top textbox dictionary
    top_box = dict(textbox)  # make a shallow copy
    top_box['height'] = top_height
    # y remains the same for the top textbox
    top_box['textbox_name'] = f"{base_name}_t0"  # rename with _t0

    # Create the bottom textbox dictionary
    bottom_box = dict(textbox)  # make a shallow copy
    bottom_box['y'] = textbox['y'] + top_height  # adjust the y position
    bottom_box['height'] = bottom_height
    bottom_box['textbox_name'] = f"{base_name}_t1"  # rename with _t1

    return top_box, bottom_box

def generate_constrained_layout(paper_panels, poster_w, poster_h, title_height_ratio=0.1):
    # Find title panel explicitly
    try:
        title_panel = next(p for p in paper_panels if ('title' in p["section_name"].lower()))
        other_panels = [p for p in paper_panels if ('title' not in p["section_name"].lower())]
    except StopIteration:
        print('Oops, no title found, please try again.')
        raise

    title_h = poster_h * title_height_ratio
    title_layout = {
        "panel_name": title_panel["section_name"],
        "panel_id": title_panel["panel_id"],
        "x": 0, "y": 0,
        "width": poster_w, "height": title_h
    }

    # Generate recursive layout on remaining space for other panels
    layout_loss, remaining_layout = panel_layout_generation(
        other_panels,
        x=0, y=title_h,
        w=poster_w, h=poster_h - title_h
    )

    # Combine title panel with others
    complete_layout = [title_layout] + remaining_layout
    return layout_loss, complete_layout


def main_inference(
    paper_panels,
    panel_model_params,
    figure_model_params,
    poster_width=1200,
    poster_height=800,
    shrink_margin=0
):
    for p in paper_panels:
        sp, rp = infer_panel_attrs(panel_model_params, p["tp"], p["gp"])
        p["sp"] = sp
        p["rp"] = rp

    layout_loss, panel_arrangement = generate_constrained_layout(paper_panels, poster_width, poster_height, title_height_ratio=0.1)
    print("Panel layout cost:", layout_loss)
    for p in panel_arrangement:
        print("Panel:", p)

    panel_map = {}
    for p in paper_panels:
        panel_map[p["panel_id"]] = p

    final_panels = []
    for pa in panel_arrangement:
        # Merge bounding box with the original sp,rp data
        pid = pa["panel_id"]
        merged_panel = {
            "panel_id": pid,
            "panel_name": pa['panel_name'],
            "x": pa["x"] + shrink_margin,
            "y": pa["y"] + shrink_margin,
            "width": pa["width"] - 2 * shrink_margin,
            "height": pa["height"] - 2 * shrink_margin,
            "sp": panel_map[pid]["sp"],
            "rp": panel_map[pid]["rp"],
            "text_len": panel_map[pid]["text_len"],
            "figure_size": panel_map[pid]["figure_size"],
            "figure_aspect": panel_map[pid]["figure_aspect"]
        }
        final_panels.append(merged_panel)

    text_arrangement = []
    figure_arrangement = []

    for p in final_panels:
        text_boxes, fig_boxes = place_text_and_figures_exact(p, figure_model_params)
        text_arrangement.extend(text_boxes)          # text arrangement
        figure_arrangement.extend(fig_boxes)       # figure arrangement

    return panel_arrangement, figure_arrangement, text_arrangement

def visualize_complete_layout(
    panels, text_boxes, figure_boxes, poster_width, poster_height
):
    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_xlim(0, poster_width)
    ax.set_ylim(0, poster_height)
    ax.set_aspect('equal')

    # Draw panels
    for panel in panels:
        rect = patches.Rectangle(
            (panel["x"], panel["y"]), panel["width"], panel["height"],
            linewidth=1, edgecolor='black', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            panel["x"] + 5, panel["y"] + panel["height"] - 5,
            f'Panel {panel["panel_id"]}', fontsize=8, va='top', color='black'
        )

    # Draw text boxes
    for txt in text_boxes:
        rect = patches.Rectangle(
            (txt["x"], txt["y"]), txt["width"], txt["height"],
            linewidth=1, edgecolor='green', linestyle='-.', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            txt["x"] + 2, txt["y"] + txt["height"] - 2,
            f'Text {txt["panel_id"]}', fontsize=7, color='green', va='top'
        )

    # Draw figures
    for fig_box in figure_boxes:
        rect = patches.Rectangle(
            (fig_box["x"], fig_box["y"]), fig_box["width"], fig_box["height"],
            linewidth=1, edgecolor='blue', linestyle='--', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            fig_box["x"] + 2, fig_box["y"] + 2,
            f'Fig {fig_box["panel_id"]}', fontsize=7, color='blue', va='bottom'
        )

    plt.gca().invert_yaxis()  # optional: invert y-axis if needed
    plt.show()


def get_arrangments_in_inches(
    width, 
    height, 
    panel_arrangement, 
    figure_arrangement, 
    text_arrangement,
    units_per_inch=72
):

    panel_arrangement_inches = copy.deepcopy(panel_arrangement)
    figure_arrangement_inches = copy.deepcopy(figure_arrangement)
    text_arrangement_inches = copy.deepcopy(text_arrangement)

    for p in panel_arrangement_inches:
        p["x"] = to_inches(p["x"], units_per_inch)
        p["y"] = to_inches(p["y"], units_per_inch)
        p["width"] = to_inches(p["width"], units_per_inch)
        p["height"] = to_inches(p["height"], units_per_inch)

    for f in figure_arrangement_inches:
        f["x"] = to_inches(f["x"], units_per_inch)
        f["y"] = to_inches(f["y"], units_per_inch)
        f["width"] = to_inches(f["width"], units_per_inch)
        f["height"] = to_inches(f["height"], units_per_inch)

    for t in text_arrangement_inches:
        t["x"] = to_inches(t["x"], units_per_inch)
        t["y"] = to_inches(t["y"], units_per_inch)
        t["width"] = to_inches(t["width"], units_per_inch)
        t["height"] = to_inches(t["height"], units_per_inch)

    width_inch, height_inch = to_inches(width, units_per_inch), to_inches(height, units_per_inch)
    return width_inch, height_inch, panel_arrangement_inches, figure_arrangement_inches, text_arrangement_inches