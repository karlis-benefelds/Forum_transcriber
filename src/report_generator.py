import json
import csv
import datetime
import re
import iso8601
from datetime import timezone
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from src.utils import _fmt_mmss, _safe_date, normalize_sentence_spacing, soft_break_long_token, get_temp_path

def _build_speaker_window_map(events_data, privacy_mode: str):
    """
    Build a map of (start,end) -> display_name based on privacy mode (names/ids).
    """
    speaker_map = {}
    for ev in events_data.get('voice_events', []):
        fn = (ev['speaker'].get('first_name') or '').strip()
        ln = (ev['speaker'].get('last_name') or '').strip()
        uid = ev['speaker'].get('id')
        name = (f"{fn} {ln}".strip() or "Professor")
        if privacy_mode == "ids" and uid:
            disp = f"ID {uid}"
        else:
            disp = name
        speaker_map[(ev['start'], ev['end'])] = disp
    return speaker_map

def compile_transcript_to_pdf(class_id, headers, privacy_mode="names"):
    try:
        # Load JSONs
        with open(get_temp_path(f"session_{class_id}_transcript.json"), 'r') as f:
            transcript_data = json.load(f)
        with open(get_temp_path(f"session_{class_id}_events.json"), 'r') as f:
            events_data = json.load(f)

        class_meta = events_data.get('class_meta', {})
        timeline_segments = events_data.get('timeline_segments', [])
        attendance = events_data.get('attendance', [])

        # Speaker mapping
        speaker_map = _build_speaker_window_map(events_data, privacy_mode)

        def find_speaker_at_time(time_point):
            for (start, end), speaker in speaker_map.items():
                if start <= time_point <= end:
                    return speaker
            return "Professor"  # generic fallback

        # Combine consecutive segments by same speaker
        compiled_entries = []
        current_entry = {'speaker': None, 'start_time': None, 'text': [], 'end_time': None}
        for segment in transcript_data['segments']:
            start_time = segment['start']; end_time = segment['end']
            raw_text = (segment['text'] or "").strip()
            if not raw_text:
                continue
            current_speaker = find_speaker_at_time(start_time)

            start_new = False
            if not current_entry['speaker']:
                start_new = True
            elif current_entry['speaker'] != current_speaker:
                start_new = True
            elif current_entry['end_time'] is not None and start_time - current_entry['end_time'] > 2:
                start_new = True

            if start_new:
                if current_entry['speaker']:
                    compiled_entries.append(current_entry)
                current_entry = {
                    'speaker': current_speaker,
                    'start_time': start_time,
                    'text': [raw_text],
                    'end_time': end_time
                }
            else:
                current_entry['text'].append(raw_text)
                current_entry['end_time'] = end_time
        if current_entry['speaker']:
            compiled_entries.append(current_entry)

        # Styles
        styles = getSampleStyleSheet()
        contribution_style = ParagraphStyle('ContributionStyle', parent=styles['Normal'],
                                            fontName='Helvetica', fontSize=10, leading=12, wordWrap='CJK')
        header_style = ParagraphStyle('HeaderStyle', parent=styles['Normal'], fontName='Helvetica-Bold',
                                      fontSize=12, textColor=colors.whitesmoke, alignment=1)
        speaker_style = ParagraphStyle('SpeakerStyle', parent=styles['Normal'],
                                       fontName='Helvetica', fontSize=10, leading=12, wordWrap='CJK')

        # PDF - save to outputs directory
        suffix = "names" if privacy_mode == "names" else "ids"
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        output_path = outputs_dir / f"session_{class_id}_transcript_{suffix}.pdf"
        doc = SimpleDocTemplate(str(output_path), pagesize=letter, rightMargin=60, leftMargin=60, topMargin=60, bottomMargin=60)

        elements = []

        # ====== HEADER / TITLE ======
        session_line = class_meta.get('session_title') or f"Session {class_id}"
        elements.append(Paragraph(session_line, styles['Title']))

        # Centered schedule line (sec_sched)
        sec_sched = class_meta.get('section_title', '') or class_meta.get('schedule', '')
        if sec_sched:
            centered_info_style = ParagraphStyle('CenteredInfo', parent=styles['Heading3'], alignment=1)
            elements.append(Paragraph(sec_sched, centered_info_style))
        elements.append(Spacer(1, 10))

        # Left-aligned meta lines (bold labels)
        # Class ID, Class Date/Time from recording_start (UTC), Class Link
        rec_start = class_meta.get('recording_start')
        dt_str = ""
        if rec_start:
            try:
                dt = iso8601.parse_date(rec_start).astimezone(timezone.utc)
                dt_str = dt.strftime("%Y-%m-%d %H:%M UTC")
            except:
                dt_str = _safe_date(rec_start)
        meta_lines = []
        meta_lines.append(Paragraph(f"<b>Class ID:</b> {class_meta.get('class_id','')}", styles['Normal']))
        if dt_str:
            meta_lines.append(Paragraph(f"<b>Class Date/Time:</b> {dt_str}", styles['Normal']))
        link = class_meta.get('class_link') or ""
        if link:
            meta_lines.append(Paragraph(f"<b>Class Link:</b> {link}", styles['Normal']))
        for p in meta_lines:
            elements.append(p)
        elements.append(Spacer(1, 14))

        # ====== ATTENDANCE TABLE ======
        if attendance:
            elements.append(Paragraph("Attendance", styles['Heading3']))
            att_rows = [[Paragraph('Student', header_style), Paragraph('Status', header_style)]]
            for a in attendance:
                display_name = a.get('name','')
                if privacy_mode == "ids" and a.get('id'):
                    display_name = f"ID {a['id']}"
                att_rows.append([
                    Paragraph(soft_break_long_token(display_name, 14), speaker_style),
                    Paragraph("Absent" if a.get('absent') else "Present", speaker_style)
                ])
            att_table = Table(att_rows, colWidths=[4.5*inch, 1.5*inch], repeatRows=1)
            att_style = TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,0), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
                ('FONTSIZE', (0,1), (-1,-1), 10),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
                ('RIGHTPADDING', (0,0), (-1,-1), 6),
                ('TOPPADDING', (0,0), (-1,-1), 3),
                ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ])
            for i, a in enumerate(attendance, start=1):
                color = colors.red if a.get('absent') else colors.green
                att_style.add('TEXTCOLOR', (1,i), (1,i), color)
            att_table.setStyle(att_style)
            elements.append(att_table)
            elements.append(Spacer(1, 16))

        # ====== CLASS EVENTS TABLE ======
        if timeline_segments:
            elements.append(Paragraph("Class Events", styles['Heading3']))
            events_data_rows = [[Paragraph('Time', header_style),
                                 Paragraph('Section', header_style),
                                 Paragraph('Event', header_style)]]
            for seg in timeline_segments:
                events_data_rows.append([
                    _fmt_mmss(seg.get('offset_seconds')),
                    Paragraph((seg.get('section','') or ''), styles['Normal']),
                    Paragraph((seg.get('title','') or ''), styles['Normal']),
                ])
            # Adjusted widths to reduce overlap, text wrapped via Paragraph
            events_table = Table(events_data_rows, colWidths=[0.9*inch, 2.2*inch, 3.9*inch], repeatRows=1)
            events_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,0), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
                ('FONTSIZE', (0,1), (-1,-1), 10),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
                ('RIGHTPADDING', (0,0), (-1,-1), 6),
                ('TOPPADDING', (0,0), (-1,-1), 3),
                ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ]))
            elements.append(events_table)
            elements.append(Spacer(1, 16))

        # ====== TRANSCRIPT (break only by class events) ======
        elements.append(Paragraph("Transcript", styles['Heading3']))
        elements.append(Spacer(1, 6))

        # Flatten and normalize
        all_items = []
        for entry in compiled_entries:
            text = normalize_sentence_spacing(' '.join(entry['text']).strip())
            if text in ['...', '.', '', 'Mm-hmm.'] or len(text) < 3:
                continue
            timestamp = _fmt_mmss(entry['start_time'])
            # split into printable chunks to avoid super long cells
            max_chars_per_chunk = 500
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks, curr = [], ""
            for s in sentences:
                candidate = (curr + s + " ").strip() if curr else (s + " ")
                if len(candidate) <= max_chars_per_chunk:
                    curr = candidate
                else:
                    if curr: chunks.append(curr.strip())
                    curr = s + " "
            if curr: chunks.append(curr.strip())

            for i, chunk in enumerate(chunks or [text]):
                display_ts = "(cont.)" if i > 0 else timestamp
                all_items.append({
                    'start_time': entry['start_time'],
                    'end_time': entry['end_time'],
                    'timestamp': display_ts,
                    'speaker': entry['speaker'],
                    'text': chunk
                })
        all_items.sort(key=lambda x: x['start_time'])

        # Build event windows; include preamble
        seg_windows = []
        if timeline_segments:
            first_start = max(0, (timeline_segments[0].get('offset_seconds') or 0))
            if first_start > 0:
                seg_windows.append({'start': 0, 'end': first_start, 'label': f"{_fmt_mmss(0)} — Before first event"})
            for idx, seg in enumerate(timeline_segments):
                start = max(0, (seg.get('offset_seconds') or 0))
                end = (timeline_segments[idx+1].get('offset_seconds') if idx+1 < len(timeline_segments) else float('inf')) or float('inf')
                bits = []
                if seg.get('section'): bits.append(seg['section'])
                if seg.get('title'):   bits.append(seg['title'])
                label = f"{_fmt_mmss(start)} — " + (' · '.join(bits) if bits else 'Event')
                seg_windows.append({'start': start, 'end': end, 'label': label})
        else:
            seg_windows.append({'start': 0, 'end': float('inf'), 'label': "Transcript"})

        for win in seg_windows:
            bucket = [it for it in all_items if win['start'] <= it['start_time'] < win['end']]
            if not bucket:
                continue
            elements.append(Paragraph(win['label'], styles['Heading4']))
            elements.append(Spacer(1, 4))

            data = [[Paragraph('Time', header_style),
                     Paragraph('Speaker', header_style),
                     Paragraph('Contribution', header_style)]]
            for item in bucket:
                spk_txt = soft_break_long_token(item['speaker'], 14)
                data.append([
                    item['timestamp'],
                    Paragraph(spk_txt, speaker_style),
                    Paragraph(normalize_sentence_spacing(item['text']), contribution_style)
                ])

            table = Table(data, colWidths=[0.75*inch, 2.1*inch, 4.25*inch], repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 12))

        doc.build(elements)
        print(f"Created PDF transcript: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error processing transcript: {str(e)}")
        raise e

def compile_transcript_to_csv(class_id, headers, privacy_mode="names"):
    """
    CSV:
      - Header block (centered items as lines)
      - Attendance
      - Class Events
      - Transcript with event headers
    """
    try:
        with open(get_temp_path(f"session_{class_id}_transcript.json"), 'r') as f:
            transcript_data = json.load(f)
        with open(get_temp_path(f"session_{class_id}_events.json"), 'r') as f:
            events_data = json.load(f)

        class_meta = events_data.get('class_meta', {})
        timeline_segments = events_data.get('timeline_segments', [])
        attendance = events_data.get('attendance', [])

        # Speaker mapping
        speaker_map = _build_speaker_window_map(events_data, privacy_mode)

        def find_speaker_at_time(t):
            for (start, end), spk in speaker_map.items():
                if start <= t <= end:
                    return spk
            return "Professor"

        # Combine segments by speaker
        compiled_entries = []
        current = {'speaker': None, 'start_time': None, 'text': [], 'end_time': None}
        for seg in transcript_data['segments']:
            st, en, tx = seg['start'], seg['end'], (seg['text'] or '').strip()
            if not tx:
                continue
            spk = find_speaker_at_time(st)

            start_new = (not current['speaker']) or (current['speaker'] != spk) or (current['end_time'] is not None and st - current['end_time'] > 2)
            if start_new:
                if current['speaker']:
                    compiled_entries.append(current)
                current = {'speaker': spk, 'start_time': st, 'text': [tx], 'end_time': en}
            else:
                current['text'].append(tx)
                current['end_time'] = en
        if current['speaker']:
            compiled_entries.append(current)

        # Flatten
        all_items = []
        for entry in compiled_entries:
            text = normalize_sentence_spacing(' '.join(entry['text']).strip())
            if text in ['...', '.', '', 'Mm-hmm.'] or len(text) < 3:
                continue
            timestamp = _fmt_mmss(entry['start_time'])
            all_items.append({
                'timestamp': timestamp,
                'speaker': entry['speaker'],
                'text': text,
                'start_time': entry['start_time'],
                'end_time': entry['end_time']
            })
        all_items.sort(key=lambda x: x['start_time'])

        # Build event windows
        segmented_rows = []
        seg_windows = []
        if timeline_segments:
            first_start = max(0, (timeline_segments[0].get('offset_seconds') or 0))
            if first_start > 0:
                seg_windows.append({'start': 0, 'end': first_start, 'label': f"{_fmt_mmss(0)} — Before first event"})
            for idx, seg in enumerate(timeline_segments):
                start = max(0, (seg.get('offset_seconds') or 0))
                end = (timeline_segments[idx+1].get('offset_seconds') if idx+1 < len(timeline_segments) else float('inf')) or float('inf')
                bits = []
                if seg.get('section'): bits.append(seg['section'])
                if seg.get('title'):   bits.append(seg['title'])
                label = f"{_fmt_mmss(start)} — " + (' / '.join(bits) if bits else 'Event')
                seg_windows.append({'start': start, 'end': end, 'label': label})
        else:
            seg_windows.append({'start': 0, 'end': float('inf'), 'label': "Transcript"})

        for win in seg_windows:
            bucket = [it for it in all_items if win['start'] <= it['start_time'] < win['end']]
            if not bucket:
                continue
            segmented_rows.append({'timestamp': '', 'speaker': '', 'text': f"--- {win['label']} ---"})
            segmented_rows.extend(bucket)

        out_rows = segmented_rows or all_items

        # Write CSV - save to outputs directory
        suffix = "names" if privacy_mode == "names" else "ids"
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        output_path = outputs_dir / f"session_{class_id}_transcript_{suffix}.csv"
        with open(str(output_path), 'w', newline='', encoding='utf-8') as csvfile:
            w = csv.writer(csvfile)

            # Header block
            w.writerow(["Session", class_meta.get('session_title','')])
            sec_sched = class_meta.get('section_title') or class_meta.get('schedule') or ''
            if sec_sched:
                w.writerow([sec_sched])
            # Bold not supported in CSV, but include meta lines:
            w.writerow(["Class ID", class_meta.get('class_id','')])
            rec_start = class_meta.get('recording_start')
            dt_str = ""
            if rec_start:
                try:
                    dt = iso8601.parse_date(rec_start).astimezone(timezone.utc)
                    dt_str = dt.strftime("%Y-%m-%d %H:%M UTC")
                except:
                    dt_str = _safe_date(rec_start)
            if dt_str:
                w.writerow(["Class Date/Time", dt_str])
            link = class_meta.get('class_link') or ""
            if link:
                w.writerow(["Class Link", link])
            w.writerow([])

            # Attendance
            if attendance:
                w.writerow(["Attendance"])
                w.writerow(["Student", "Status"])
                for a in attendance:
                    display_name = a.get('name','')
                    if privacy_mode == "ids" and a.get('id'):
                        display_name = f"ID {a['id']}"
                    w.writerow([display_name, "Absent" if a.get('absent') else "Present"])
                w.writerow([])

            # Class Events
            if timeline_segments:
                w.writerow(["Class Events"])
                w.writerow(["Time", "Section", "Event"])
                for seg in timeline_segments:
                    w.writerow([_fmt_mmss(seg.get('offset_seconds')), seg.get('section',''), seg.get('title','')])
                w.writerow([])

            # Transcript
            w.writerow(['Time', 'Speaker', 'Contribution'])
            for row in out_rows:
                w.writerow([row.get('timestamp',''), row.get('speaker',''), row.get('text','')])

        print(f"Created CSV transcript: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error creating CSV transcript: {str(e)}")
        return None

def create_simplified_csv(class_id, transcript_path):
    """
    Fallback CSV: just time + text.
    """
    try:
        with open(get_temp_path(f"session_{class_id}_transcript.json"), 'r') as f:
            transcript_data = json.load(f)

        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        output_path = outputs_dir / f"session_{class_id}_transcript_simple.csv"
        with open(str(output_path), 'w', newline='', encoding='utf-8') as csvfile:
            w = csv.writer(csvfile)
            w.writerow(['Time', 'Text'])
            for seg in transcript_data['segments']:
                minutes = int(seg['start'] // 60)
                seconds = int(seg['start'] % 60)
                timestamp = f"{minutes:02d}:{seconds:02d}"
                w.writerow([timestamp, normalize_sentence_spacing(seg['text'])])

        print(f"Created simplified CSV transcript: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error creating simplified CSV transcript: {str(e)}")
        return None

def create_simplified_transcript(class_id, transcript_path):
    """
    Fallback PDF: no events, no speakers; includes minimal title.
    """
    try:
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        output_path = outputs_dir / f"session_{class_id}_transcript_simple.pdf"
        styles = getSampleStyleSheet()
        text_style = ParagraphStyle('TextStyle', parent=styles['Normal'],
                                    fontName='Helvetica', fontSize=10, leading=12, spaceAfter=0, spaceBefore=0,
                                    wordWrap='CJK')
        header_style = ParagraphStyle('HeaderStyle', parent=styles['Normal'],
                                      fontName='Helvetica-Bold', fontSize=12, textColor=colors.whitesmoke,
                                      alignment=1)

        with open(get_temp_path(f"session_{class_id}_transcript.json"), 'r') as f:
            transcript_data = json.load(f)

        doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                                rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        elements = []
        title = Paragraph(f"Session {class_id}", styles['Title'])
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        subtitle = Paragraph(f"Generated on {date_str}", styles['Heading2'])
        elements.append(title); elements.append(subtitle); elements.append(Spacer(1, 12))

        data = [[Paragraph('Time', header_style), Paragraph('Text', header_style)]]
        for seg in transcript_data['segments']:
            minutes = int(seg['start'] // 60)
            seconds = int(seg['start'] % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"
            data.append([timestamp, Paragraph(normalize_sentence_spacing(seg['text']), text_style)])

        table = Table(data, colWidths=[0.75*inch, 6.25*inch], repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 3),
        ]))
        elements.append(table)
        doc.build(elements)
        print(f"Created simplified PDF transcript: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error creating simplified transcript: {str(e)}")
        return None
