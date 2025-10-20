from PosterAgent.parse_raw import parse_raw, gen_image_and_table
from PosterAgent.gen_outline_layout import filter_image_table, gen_outline_layout_v2
from utils.wei_utils import get_agent_config, utils_functions, run_code, scale_to_target_area, char_capacity
from PosterAgent.tree_split_layout import main_train, main_inference, get_arrangments_in_inches, split_textbox, to_inches
from PosterAgent.gen_pptx_code import generate_poster_code
from utils.src.utils import ppt_to_images
from PosterAgent.gen_poster_content import gen_bullet_point_content
from utils.ablation_utils import no_tree_get_layout

# Import refactored utilities
from utils.logo_utils import LogoManager, add_logos_to_poster_code
from utils.config_utils import (
    load_poster_yaml_config, extract_font_sizes, extract_colors,
    extract_vertical_alignment, extract_section_title_symbol, normalize_config_values
)
from utils.style_utils import apply_all_styles
from utils.theme_utils import get_default_theme, create_theme_with_alignment, resolve_colors

import argparse
import json
import os
import time
import shutil

units_per_inch = 25

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Poster Generation Pipeline with Logo Support')
    parser.add_argument('--poster_path', type=str)
    parser.add_argument('--model_name_t', type=str, default='4o')
    parser.add_argument('--model_name_v', type=str, default='4o')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--poster_name', type=str, default=None)
    parser.add_argument('--tmp_dir', type=str, default='tmp')
    parser.add_argument('--estimate_chars', action='store_true')
    parser.add_argument('--max_workers', type=int, default=10)
    parser.add_argument('--poster_width_inches', type=int, default=None)
    parser.add_argument('--poster_height_inches', type=int, default=None)
    parser.add_argument('--no_blank_detection', action='store_true', help='When overflow is severe, try this option.')
    parser.add_argument('--ablation_no_tree_layout', action='store_true', help='Ablation study: no tree layout')
    parser.add_argument('--ablation_no_commenter', action='store_true', help='Ablation study: no commenter')
    parser.add_argument('--ablation_no_example', action='store_true', help='Ablation study: no example')

    # Logo-related arguments
    parser.add_argument('--conference_venue', type=str, default=None,
                       help='Conference name for automatic logo search (e.g., "NeurIPS", "CVPR")')
    parser.add_argument('--institution_logo_path', type=str, default=None,
                       help='Custom path to institution logo (auto-searches from paper metadata if not provided)')
    parser.add_argument('--conference_logo_path', type=str, default=None,
                       help='Custom path to conference logo (auto-searches if venue specified)')
    parser.add_argument('--use_google_search', action='store_true',
                       help='Use Google Custom Search API for logo search (requires API keys in .env)')

    args = parser.parse_args()

    start_time = time.time()

    os.makedirs(args.tmp_dir, exist_ok=True)

    detail_log = {}

    agent_config_t = get_agent_config(args.model_name_t)
    agent_config_v = get_agent_config(args.model_name_v)
    poster_name = args.poster_path.split('/')[-2].replace(' ', '_')
    if args.poster_name is None:
        args.poster_name = poster_name
    else:
        poster_name = args.poster_name
    meta_json_path = args.poster_path.replace('paper.pdf', 'meta.json')
    if args.poster_width_inches is not None and args.poster_height_inches is not None:
        poster_width = args.poster_width_inches * units_per_inch
        poster_height = args.poster_height_inches * units_per_inch
    elif os.path.exists(meta_json_path):
        meta_json = json.load(open(meta_json_path, 'r'))
        poster_width = meta_json['width']
        poster_height = meta_json['height']
    else:
        poster_width = 48 * units_per_inch
        poster_height = 36 * units_per_inch

    poster_width, poster_height = scale_to_target_area(poster_width, poster_height)
    poster_width_inches = to_inches(poster_width, units_per_inch)
    poster_height_inches = to_inches(poster_height, units_per_inch)

    if poster_width_inches > 56 or poster_height_inches > 56:
        # Work out which side is longer, then compute a single scale factor
        if poster_width_inches >= poster_height_inches:
            scale_factor = 56 / poster_width_inches
        else:
            scale_factor = 56 / poster_height_inches

        poster_width_inches  *= scale_factor
        poster_height_inches *= scale_factor

        # convert back to internal units
        poster_width  = poster_width_inches  * units_per_inch
        poster_height = poster_height_inches * units_per_inch

    print(f'Poster size: {poster_width_inches} x {poster_height_inches} inches')

    total_input_tokens_t, total_output_tokens_t = 0, 0
    total_input_tokens_v, total_output_tokens_v = 0, 0

    # Step 1: Parse the raw poster
    input_token, output_token, raw_result = parse_raw(args, agent_config_t, version=2)
    total_input_tokens_t += input_token
    total_output_tokens_t += output_token

    _, _, images, tables = gen_image_and_table(args, raw_result)

    print(f'Parsing token consumption: {input_token} -> {output_token}')

    parser_time_taken = time.time() - start_time
    print(f'Parser time: {parser_time_taken:.2f} seconds')
    detail_log['parser_time'] = parser_time_taken

    parser_time = time.time()

    detail_log['parser_in_t'] = input_token
    detail_log['parser_out_t'] = output_token

    # Initialize LogoManager
    logo_manager = LogoManager()
    institution_logo_path = args.institution_logo_path
    conference_logo_path = args.conference_logo_path

    # Auto-detect institution from paper if not provided
    # Now using the raw_result directly instead of reading from file
    if not institution_logo_path:
        print("\n" + "="*60)
        print("🔍 AUTO-DETECTING INSTITUTION FROM PAPER")
        print("="*60)

        # Use the raw_result we already have from the parser
        if raw_result:
            print(f"📄 Using parsed paper content")
            # Extract text content from the ConversionResult object
            try:
                paper_text = raw_result.document.export_to_markdown()
            except:
                # Fallback: try to get text content in another way
                paper_text = str(raw_result)

            print("🔎 Searching for FIRST AUTHOR's institution...")
            first_author_inst = logo_manager.extract_first_author_institution(paper_text)

            if first_author_inst:
                print(f"\n✅ FIRST AUTHOR INSTITUTION: {first_author_inst}")
                print(f"🔍 Searching for logo: {first_author_inst}")

                inst_logo_path = logo_manager.get_logo_path(first_author_inst, category="institute", use_google=args.use_google_search)
                if inst_logo_path:
                    institution_logo_path = str(inst_logo_path)
                    print(f"✅ Institution logo found: {institution_logo_path}")
                else:
                    print(f"❌ Could not find/download logo for: {first_author_inst}")
            else:
                print("❌ No first author institution detected or matched with available logos")
        else:
            print("❌ No parsed content available")
        print("="*60 + "\n")

    # Handle conference logo
    if args.conference_venue and not conference_logo_path:
        print("\n" + "="*60)
        print("🏛️ SEARCHING FOR CONFERENCE LOGO")
        print("="*60)
        print(f"📍 Conference: {args.conference_venue}")
        print(f"🔍 Searching for logo...")

        conf_logo_path = logo_manager.get_logo_path(args.conference_venue, category="conference", use_google=args.use_google_search)
        if conf_logo_path:
            conference_logo_path = str(conf_logo_path)
            print(f"✅ Conference logo found: {conference_logo_path}")
        else:
            print(f"❌ Could not find/download logo for: {args.conference_venue}")
            # Note: Web search is now handled inside get_logo_path automatically
        print("="*60 + "\n")

    # Step 2: Filter unnecessary images and tables
    input_token, output_token = filter_image_table(args, agent_config_t)
    total_input_tokens_t += input_token
    total_output_tokens_t += output_token
    print(f'Filter figures token consumption: {input_token} -> {output_token}')

    filter_time_taken = time.time() - parser_time
    print(f'Filter time: {filter_time_taken:.2f} seconds')
    detail_log['filter_time'] = filter_time_taken

    filter_time = time.time()

    detail_log['filter_in_t'] = input_token
    detail_log['filter_out_t'] = output_token

    # Step 3: Generate outline
    input_token, output_token, panels, figures = gen_outline_layout_v2(args, agent_config_t)
    total_input_tokens_t += input_token
    total_output_tokens_t += output_token
    print(f'Outline token consumption: {input_token} -> {output_token}')

    outline_time_taken = time.time() - filter_time
    print(f'Outline time: {outline_time_taken:.2f} seconds')
    detail_log['outline_time'] = outline_time_taken

    outline_time = time.time()

    detail_log['outline_in_t'] = input_token
    detail_log['outline_out_t'] = output_token

    if args.ablation_no_tree_layout:
        panel_arrangement, figure_arrangement, text_arrangement, input_token, output_token = no_tree_get_layout(
            poster_width,
            poster_height,
            panels,
            figures,
            agent_config_t
        )
        total_input_tokens_t += input_token
        total_output_tokens_t += output_token
        print(f'No tree layout token consumption: {input_token} -> {output_token}')
        detail_log['no_tree_layout_in_t'] = input_token
        detail_log['no_tree_layout_out_t'] = output_token
    else:

        # Step 4: Learn and generate layout
        panel_model_params, figure_model_params = main_train()

        panel_arrangement, figure_arrangement, text_arrangement = main_inference(
            panels,
            panel_model_params,
            figure_model_params,
            poster_width,
            poster_height,
            shrink_margin=3
        )

        text_arrangement_title = text_arrangement[0]
        text_arrangement = text_arrangement[1:]
        # Split the title textbox into two parts
        text_arrangement_title_top, text_arrangement_title_bottom = split_textbox(
            text_arrangement_title,
            0.8
        )
        # Add the split textboxes back to the list
        text_arrangement = [text_arrangement_title_top, text_arrangement_title_bottom] + text_arrangement

    for i in range(len(figure_arrangement)):
        panel_id = figure_arrangement[i]['panel_id']
        panel_section_name = panels[panel_id]['section_name']
        figure_info = figures[panel_section_name]
        if 'image' in figure_info:
            figure_id = figure_info['image']
            if not figure_id in images:
                figure_path = images[str(figure_id)]['image_path']
            else:
                figure_path = images[figure_id]['image_path']
        elif 'table' in figure_info:
            figure_id = figure_info['table']
            if not figure_id in tables:
                figure_path = tables[str(figure_id)]['table_path']
            else:
                figure_path = tables[figure_id]['table_path']

        figure_arrangement[i]['figure_path'] = figure_path

    for text_arrangement_item in text_arrangement:
        num_chars = char_capacity(
            bbox=(text_arrangement_item['x'], text_arrangement_item['y'], text_arrangement_item['height'], text_arrangement_item['width'])
        )
        text_arrangement_item['num_chars'] = num_chars


    width_inch, height_inch, panel_arrangement_inches, figure_arrangement_inches, text_arrangement_inches = get_arrangments_in_inches(
        poster_width, poster_height, panel_arrangement, figure_arrangement, text_arrangement, 25
    )

    # Save to file
    tree_split_results = {
        'poster_width': poster_width,
        'poster_height': poster_height,
        'poster_width_inches': width_inch,
        'poster_height_inches': height_inch,
        'panels': panels,
        'panel_arrangement': panel_arrangement,
        'figure_arrangement': figure_arrangement,
        'text_arrangement': text_arrangement,
        'panel_arrangement_inches': panel_arrangement_inches,
        'figure_arrangement_inches': figure_arrangement_inches,
        'text_arrangement_inches': text_arrangement_inches,
    }
    os.makedirs('tree_splits', exist_ok=True)
    with open(f'tree_splits/<{args.model_name_t}_{args.model_name_v}>_{args.poster_name}_tree_split_{args.index}.json', 'w') as f:
        json.dump(tree_split_results, f, indent=4)

    layout_time_taken = time.time() - outline_time
    print(f'Layout time: {layout_time_taken:.2f} seconds')
    detail_log['layout_time'] = layout_time_taken

    layout_time = time.time()

    # === Configuration Loading ===
    print("\n📋 Loading configuration from YAML files...", flush=True)
    yaml_cfg = load_poster_yaml_config(args.poster_path)

    # Extract configuration values
    bullet_fs, title_fs, poster_title_fs, poster_author_fs = extract_font_sizes(yaml_cfg)
    title_text_color, title_fill_color, main_text_color, main_text_fill_color = extract_colors(yaml_cfg)
    section_title_vertical_align = extract_vertical_alignment(yaml_cfg)
    section_title_symbol = extract_section_title_symbol(yaml_cfg)

    # Normalize configuration values
    bullet_fs, title_fs, poster_title_fs, poster_author_fs, \
    title_text_color, title_fill_color, main_text_color, main_text_fill_color = normalize_config_values(
        bullet_fs, title_fs, poster_title_fs, poster_author_fs,
        title_text_color, title_fill_color, main_text_color, main_text_fill_color
    )

    # Store configuration in args
    setattr(args, 'bullet_font_size', bullet_fs)
    setattr(args, 'section_title_font_size', title_fs)
    setattr(args, 'poster_title_font_size', poster_title_fs)
    setattr(args, 'poster_author_font_size', poster_author_fs)
    setattr(args, 'title_text_color', title_text_color)
    setattr(args, 'title_fill_color', title_fill_color)
    setattr(args, 'main_text_color', main_text_color)
    setattr(args, 'main_text_fill_color', main_text_fill_color)
    setattr(args, 'section_title_vertical_align', section_title_vertical_align)

    # Step 5: Generate content
    print(f"\n✍️ Generating poster content (max_workers={args.max_workers})...", flush=True)
    input_token_t, output_token_t, input_token_v, output_token_v = gen_bullet_point_content(args, agent_config_t, agent_config_v, tmp_dir=args.tmp_dir)
    total_input_tokens_t += input_token
    total_output_tokens_t += output_token
    total_input_tokens_v += input_token_v
    total_output_tokens_v += output_token_v
    print(f'Content generation token consumption T: {input_token_t} -> {output_token_t}')
    print(f'Content generation token consumption V: {input_token_v} -> {output_token_v}')

    content_time_taken = time.time() - layout_time
    print(f'Content generation time: {content_time_taken:.2f} seconds')
    detail_log['content_time'] = content_time_taken

    content_time = time.time()

    bullet_content = json.load(open(f'contents/<{args.model_name_t}_{args.model_name_v}>_{args.poster_name}_bullet_point_content_{args.index}.json', 'r'))

    detail_log['content_in_t'] = input_token_t
    detail_log['content_out_t'] = output_token_t
    detail_log['content_in_v'] = input_token_v
    detail_log['content_out_v'] = output_token_v

    # === Style Application ===
    print("\n🎨 Applying styles and colors...", flush=True)

    # Resolve colors with fallbacks
    final_title_text_color, final_title_fill_color, final_main_text_color, final_main_text_fill_color = resolve_colors(
        getattr(args, 'title_text_color', None),
        getattr(args, 'title_fill_color', None),
        getattr(args, 'main_text_color', None),
        getattr(args, 'main_text_fill_color', None)
    )

    # Apply all styles in one go
    bullet_content = apply_all_styles(
        bullet_content,
        title_text_color=final_title_text_color,
        title_fill_color=final_title_fill_color,
        main_text_color=final_main_text_color,
        main_text_fill_color=final_main_text_fill_color,
        section_title_symbol=section_title_symbol,
        main_text_font_size=bullet_fs
    )

    # === Poster Generation ===
    print("\n🎯 Generating PowerPoint code...", flush=True)

    # Create theme with alignment
    base_theme = get_default_theme()
    theme_with_alignment = create_theme_with_alignment(
        base_theme,
        getattr(args, 'section_title_vertical_align', None)
    )

    poster_code = generate_poster_code(
        panel_arrangement_inches,
        text_arrangement_inches,
        figure_arrangement_inches,
        presentation_object_name='poster_presentation',
        slide_object_name='poster_slide',
        utils_functions=utils_functions,
        slide_width=width_inch,
        slide_height=height_inch,
        img_path=None,
        save_path=f'{args.tmp_dir}/poster.pptx',
        visible=False,
        content=bullet_content,
        theme=theme_with_alignment,
        tmp_dir=args.tmp_dir,
    )

    # Add logos to the poster
    print("\n🖼️ Adding logos to poster...", flush=True)
    poster_code = add_logos_to_poster_code(
        poster_code,
        width_inch,
        height_inch,
        institution_logo_path=institution_logo_path,
        conference_logo_path=conference_logo_path
    )

    output, err = run_code(poster_code)
    if err is not None:
        raise RuntimeError(f'Error in generating PowerPoint: {err}')

    # Step 8: Create a folder in the output directory
    output_dir = f'<{args.model_name_t}_{args.model_name_v}>_generated_posters/{args.poster_path.replace("paper.pdf", "")}'
    os.makedirs(output_dir, exist_ok=True)

    # Copy logos to output directory for reference
    logos_dir = os.path.join(output_dir, 'logos')
    if institution_logo_path or conference_logo_path:
        os.makedirs(logos_dir, exist_ok=True)
        if institution_logo_path and os.path.exists(institution_logo_path):
            shutil.copy2(institution_logo_path, os.path.join(logos_dir, 'institution_logo' + os.path.splitext(institution_logo_path)[1]))
        if conference_logo_path and os.path.exists(conference_logo_path):
            shutil.copy2(conference_logo_path, os.path.join(logos_dir, 'conference_logo' + os.path.splitext(conference_logo_path)[1]))

    # Step 9: Move poster.pptx to the output directory
    pptx_path = os.path.join(output_dir, f'{poster_name}.pptx')
    os.rename(f'{args.tmp_dir}/poster.pptx', pptx_path)
    print(f'Poster PowerPoint saved to {pptx_path}')
    # Step 10: Convert the PowerPoint to images
    ppt_to_images(pptx_path, output_dir)
    print(f'Poster images saved to {output_dir}')

    end_time = time.time()
    time_taken = end_time - start_time

    render_time_taken = time.time() - content_time
    print(f'Render time: {render_time_taken:.2f} seconds')
    detail_log['render_time'] = render_time_taken

    # log
    log_file = os.path.join(output_dir, 'log.json')
    with open(log_file, 'w') as f:
        log_data = {
            'input_tokens_t': total_input_tokens_t,
            'output_tokens_t': total_output_tokens_t,
            'input_tokens_v': total_input_tokens_v,
            'output_tokens_v': total_output_tokens_v,
            'time_taken': time_taken,
            'institution_logo': institution_logo_path,
            'conference_logo': conference_logo_path,
        }
        json.dump(log_data, f, indent=4)

    detail_log_file = os.path.join(output_dir, 'detail_log.json')
    with open(detail_log_file, 'w') as f:
        json.dump(detail_log, f, indent=4)

    print(f'\nTotal time: {time_taken:.2f} seconds')
    print(f'Total text model tokens: {total_input_tokens_t} -> {total_output_tokens_t}')
    print(f'Total vision model tokens: {total_input_tokens_v} -> {total_output_tokens_v}')

    if institution_logo_path:
        print(f'Institution logo added: {institution_logo_path}')
    if conference_logo_path:
        print(f'Conference logo added: {conference_logo_path}')
