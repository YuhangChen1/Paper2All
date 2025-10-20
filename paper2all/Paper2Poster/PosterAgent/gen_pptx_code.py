import re
import json

def sanitize_for_var(name):
    # Convert any character that is not alphanumeric or underscore into underscore.
    return re.sub(r'[^0-9a-zA-Z_]+', '_', name)

def initialize_poster_code(width, height, slide_object_name, presentation_object_name, utils_functions):
    code = utils_functions
    code += fr'''
# Poster: {presentation_object_name}
{presentation_object_name} = create_poster(width_inch={width}, height_inch={height})

# Slide: {slide_object_name}
{slide_object_name} = add_blank_slide({presentation_object_name})
'''

    return code

def save_poster_code(output_file, utils_functions, presentation_object_name):
    code = utils_functions
    code = fr'''
# Save the presentation
save_presentation({presentation_object_name}, file_name="{output_file}")
'''
    return code

def generate_panel_code(panel_dict, utils_functions, slide_object_name, visible=False, theme=None):
    code = utils_functions
    raw_name = panel_dict["panel_name"]
    var_name = 'var_' + sanitize_for_var(raw_name)

    code += fr'''
# Panel: {raw_name}
{var_name} = add_textbox(
    {slide_object_name}, 
    '{var_name}', 
    {panel_dict['x']}, 
    {panel_dict['y']}, 
    {panel_dict['width']}, 
    {panel_dict['height']}, 
    text="", 
    word_wrap=True,
    font_size=40,
    bold=False,
    italic=False,
    alignment="left",
    fill_color=None,
    font_name="Arial"
)
'''

    if visible:
        if theme is None:
            code += fr'''
# Make border visible
style_shape_border({var_name}, color=(0, 0, 0), thickness=5, line_style="solid")
'''
        else:
            code += fr'''
# Make border visible
style_shape_border({var_name}, color={theme['color']}, thickness={theme['thickness']}, line_style="{theme['line_style']}")
'''
    
    return code

def generate_textbox_code(
    text_dict,
    utils_functions,
    slide_object_name,
    visible=False,
    content=None,
    theme=None,
    tmp_dir='tmp',
    is_title=False,
):
    code = utils_functions
    raw_name = text_dict["textbox_name"]
    var_name = sanitize_for_var(raw_name)

    code += fr'''
# Textbox: {raw_name}
{var_name} = add_textbox(
    {slide_object_name},
    '{var_name}',
    {text_dict['x']},
    {text_dict['y']},
    {text_dict['width']},
    {text_dict['height']},
    text="",
    word_wrap=True,
    font_size=40,
    bold=False,
    italic=False,
    alignment="left",
    fill_color=None,
    font_name="Arial"
)
'''
    if visible:
        # Extract textbox_theme from full theme if needed
        textbox_border_theme = None
        if theme is not None and isinstance(theme, dict):
            textbox_border_theme = theme.get('textbox_theme')

        if textbox_border_theme is None:
            code += fr'''
# Make border visible
style_shape_border({var_name}, color=(255, 0, 0), thickness=5, line_style="solid")
'''
        else:
            code += fr'''
# Make border visible
style_shape_border({var_name}, color={textbox_border_theme['color']}, thickness={textbox_border_theme['thickness']}, line_style="{textbox_border_theme['line_style']}")
'''

    if content is not None:
        tmp_name = f'{tmp_dir}/{var_name}_content.json'
        json.dump(content, open(tmp_name, 'w'), indent=4)

        # Determine vertical alignment
        vertical_anchor = None
        if is_title and theme is not None and 'section_title_vertical_align' in theme:
            vertical_anchor = theme['section_title_vertical_align']

        if vertical_anchor:
            code += fr'''
fill_textframe({var_name}, json.load(open('{tmp_name}', 'r')), vertical_anchor="{vertical_anchor}")
'''
        else:
            code += fr'''
fill_textframe({var_name}, json.load(open('{tmp_name}', 'r')))
'''

    return code

def generate_figure_code(figure_dict, utils_functions, slide_object_name, img_path, visible=False, theme=None):
    code = utils_functions
    raw_name = figure_dict["figure_name"]
    var_name = sanitize_for_var(raw_name)

    code += fr'''
# Figure: {raw_name}
{var_name} = add_image(
    {slide_object_name}, 
    '{var_name}', 
    {figure_dict['x']}, 
    {figure_dict['y']}, 
    {figure_dict['width']}, 
    {figure_dict['height']}, 
    image_path="{img_path}"
)
'''

    if visible:
        if theme is None:
            code += fr'''
# Make border visible
style_shape_border({var_name}, color=(0, 0, 255), thickness=5, line_style="long_dash_dot")
'''
        else:
            code += fr'''
# Make border visible
style_shape_border({var_name}, color={theme['color']}, thickness={theme['thickness']}, line_style="{theme['line_style']}")
'''
    
    return code

def generate_poster_code(
    panel_arrangement_list,
    text_arrangement_list,
    figure_arrangement_list,
    presentation_object_name,
    slide_object_name,
    utils_functions,
    slide_width,
    slide_height,
    img_path,
    save_path,
    visible=False,
    content=None,
    check_overflow=False,
    theme=None,
    tmp_dir='tmp',
):
    code = ''
    code += initialize_poster_code(slide_width, slide_height, slide_object_name, presentation_object_name, utils_functions)

    if theme is None:
        panel_visible = visible
        textbox_visible = visible
        figure_visible = visible

        panel_theme, textbox_theme, figure_theme = None, None, None
    else:
        panel_visible = theme['panel_visible']
        textbox_visible = theme['textbox_visible']
        figure_visible = theme['figure_visible']
        panel_theme = theme['panel_theme']
        textbox_theme = theme['textbox_theme']
        figure_theme = theme['figure_theme']

    for p in panel_arrangement_list:
        code += generate_panel_code(p, '', slide_object_name, panel_visible, panel_theme)

    if check_overflow:
        t = text_arrangement_list[0]
        # Pass full theme for consistency
        code += generate_textbox_code(t, '', slide_object_name, textbox_visible, content, theme, tmp_dir, is_title=False)
    else:
        all_content = []
        title_indices = set()  # Track which indices are section titles
        if content is not None:
            idx = 0
            for section_content in content:
                if 'title' in section_content:
                    all_content.append(section_content['title'])
                    title_indices.add(idx)  # Mark this index as a title
                    idx += 1
                if len(section_content) == 2:
                    all_content.append(section_content['textbox1'])
                    idx += 1
                elif len(section_content) == 3:
                    all_content.append(section_content['textbox1'])
                    all_content.append(section_content['textbox2'])
                    idx += 2
                else:
                    raise ValueError(f"Unexpected content length: {len(section_content)}")

        for i in range(len(text_arrangement_list)):
            t = text_arrangement_list[i]
            if content is not None:
                textbox_content = all_content[i]
                is_title = i in title_indices
            else:
                textbox_content = None
                is_title = False
            # Pass full theme (not textbox_theme) so vertical alignment config is available
            code += generate_textbox_code(t, '', slide_object_name, textbox_visible, textbox_content, theme, tmp_dir, is_title=is_title)

    for f in figure_arrangement_list:
        if img_path is None:
            code += generate_figure_code(f, '', slide_object_name, f['figure_path'], figure_visible, figure_theme)
        else:
            code += generate_figure_code(f, '', slide_object_name, img_path, figure_visible, figure_theme)

    code += save_poster_code(save_path, '', presentation_object_name)

    return code