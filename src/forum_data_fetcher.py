import requests
import json
import iso8601
from src.utils import extract_ids_from_curl, derive_class_link_from_curl, get_temp_path

def get_forum_events(class_id, headers, raw_curl):
    print("Fetching class and event data from Forum...")
    try:
        # Class meta
        class_url = f'https://forum.minerva.edu/api/v1/class_grader/classes/{class_id}'
        r = requests.get(class_url, headers=headers, timeout=30)
        if r.status_code != 200:
            print(f"Class data error: {r.status_code}\n{r.text[:400]}")
            raise RuntimeError(f"Failed to access class data. Status code: {r.status_code}")
        data = r.json()

        session_title = data.get('title') or f"Session {class_id}"
        section_title = (data.get('section') or {}).get('title', '')  # e.g. "Terrana, MW@09:00AM San Francisco"
        course_obj = (data.get('section') or {}).get('course') or {}
        course_code  = course_obj.get('course-code', '')
        course_title = course_obj.get('title', '')
        class_type   = data.get('type', '')

        # Recording window
        rec = (data.get('recording-sessions') or [{}])[0]
        recording_start = rec.get('recording-started')
        recording_end   = rec.get('recording-ended')

        # Guess schedule from section_title "Terrana, MW@09:00AM San Francisco"
        schedule_guess = ''
        if isinstance(section_title, str) and ',' in section_title:
            parts = [p.strip() for p in section_title.split(',', 1)]
            schedule_guess = parts[1] if len(parts) > 1 else ''

        ids = extract_ids_from_curl(raw_curl)
        course_id  = ids.get("course_id")
        section_id = ids.get("section_id")
        class_link = derive_class_link_from_curl(raw_curl, course_id, section_id, str(class_id))

        # ---- Events ----
        events_url = f'https://forum.minerva.edu/api/v1/class_grader/classes/{class_id}/class-events'
        r = requests.get(events_url, headers=headers, timeout=30)
        if r.status_code != 200:
            print(f"Class events error: {r.status_code}\n{r.text[:400]}")
            raise RuntimeError(f"Failed to access class events. Status code: {r.status_code}")

        events = r.json()
        if not isinstance(events, list):
            raise ValueError("No valid class events returned from API")

        # Parse voice & timeline
        voice_events = []
        timeline_segments = []

        if not recording_start:
            raise KeyError("No recording-started found in class data")
        ref_time = iso8601.parse_date(recording_start)

        for ev in events:
            et = ev.get('event-type')
            try:
                if et == 'voice':
                    duration_ms = (ev.get('event-data') or {}).get('duration', 0)
                    duration = duration_ms / 1000.0
                    if duration >= 1:
                        start_time = iso8601.parse_date(ev['start-time'])
                        end_time   = iso8601.parse_date(ev['end-time'])
                        actor = ev.get('actor') or {}
                        voice_events.append({
                            'start': (start_time - ref_time).total_seconds(),
                            'end': (end_time - ref_time).total_seconds(),
                            'duration': duration,
                            'speaker': {
                                'id': actor.get('id'),
                                'first_name': actor.get('first-name'),
                                'last_name':  actor.get('last-name')
                            }
                        })
                elif et == 'timeline-segment':
                    start_time = iso8601.parse_date(ev['start-time'])
                    seg = (ev.get('event-data') or {})
                    timeline_segments.append({
                        'abs_start': ev['start-time'],
                        'offset_seconds': (start_time - ref_time).total_seconds(),
                        'section': seg.get('timeline-section-title', ''),
                        'title':   seg.get('timeline-segment-title', ''),
                    })
            except KeyError:
                continue

        timeline_segments.sort(key=lambda x: x['offset_seconds'])

        # Attempt attendance from class-users (best-effort)
        attendance = []
        for cu in data.get('class-users', []):
            role = (cu.get('role') or "").lower()
            if role == 'student':
                u = cu.get('user') or {}
                name = f"{(u.get('first-name') or '').strip()} {(u.get('last-name') or '').strip()}".strip()
                uid  = u.get('id') or cu.get('user-id')
                # infer absence if possible; default = present
                abs_flag = cu.get('absent')
                if abs_flag is None:
                    att = cu.get('attended')
                    abs_flag = (False if att is None else (not bool(att)))
                attendance.append({'name': name or f"ID {uid}", 'id': uid, 'absent': bool(abs_flag)})

        # Build class meta (NO instructor field)
        # Also include class_id, course/section IDs and link for headers
        class_meta = {
            'class_id': str(class_id),
            'session_title': session_title,
            'course_code': course_code,
            'course_title': course_title,
            'section_title': section_title,   # "Terrana, MW@09:00AM San Francisco"
            'schedule': schedule_guess,
            'class_type': class_type,
            'recording_start': recording_start,
            'recording_end': recording_end,
            'course_id': course_id,
            'section_id': section_id,
            'class_link': class_link
        }

        events_data = {
            'class_id': class_id,
            'class_meta': class_meta,
            'voice_events': voice_events,
            'timeline_segments': timeline_segments,
            'attendance': attendance
        }
        temp_events_path = get_temp_path(f"session_{class_id}_events.json")
        with open(temp_events_path, 'w', encoding='utf-8') as f:
            json.dump(events_data, f, indent=2)

        print(f"Processed voice events: {len(voice_events)}; timeline segments: {len(timeline_segments)}; attendance: {len(attendance)}")
        return events_data

    except Exception as e:
        print(f"Error fetching Forum events: {e}")
        raise
