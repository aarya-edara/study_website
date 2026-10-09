from flask import Flask, render_template, request, session, redirect, url_for
from datetime import datetime, timedelta

from scheduler import generate_schedule
from dotenv import load_dotenv
from study_processor import generate_study_material

import os

load_dotenv()

print("API key found:", bool(os.getenv("OPENAI_API_KEY")))

app = Flask(__name__)

app.secret_key = "schedule-maker-secret-key"


def add_timetable_entries(
    schedule,
    start_date,
    start_time,
    end_time,
    number_of_days,
    break_minutes,
    meal_start,
    meal_end,
    unavailable_times
):
    timetable = []

    for day_number in range(number_of_days):

        current_date = start_date + timedelta(days=day_number)

        day_start = datetime.combine(current_date, start_time)
        day_end = datetime.combine(current_date, end_time)

        if day_end <= day_start:
            continue

        # Find tasks scheduled for this day
        day_tasks = sorted(
            [
                item for item in schedule
                if item["date"] == current_date
            ],
            key=lambda item: item["start"]
        )

        # Build blocked periods
        blocked = []

        if meal_start and meal_end:
            blocked.append({
                "start": datetime.combine(current_date, meal_start),
                "end": datetime.combine(current_date, meal_end),
                "type": "meal",
                "task": "Lunch / Break"
            })

        for unavailable_start, unavailable_end in unavailable_times:
            blocked.append({
                "start": datetime.combine(current_date, unavailable_start),
                "end": datetime.combine(current_date, unavailable_end),
                "type": "unavailable",
                "task": "Unavailable"
            })

        # Create a timeline from all interval boundaries
        boundaries = {day_start, day_end}

        for item in day_tasks:
            boundaries.add(max(day_start, min(day_end, item["start"])))
            boundaries.add(max(day_start, min(day_end, item["end"])))

        for item in blocked:
            boundaries.add(max(day_start, min(day_end, item["start"])))
            boundaries.add(max(day_start, min(day_end, item["end"])))

        # Include expected break endpoints after study sessions
        for item in day_tasks:
            break_end = item["end"] + timedelta(minutes=break_minutes)

            if item["end"] < day_end:
                boundaries.add(min(break_end, day_end))

        boundaries = sorted(boundaries)

        for start, end in zip(boundaries, boundaries[1:]):

            if start >= end:
                continue

            # Existing scheduled task
            matching_task = next(
                (
                    item for item in day_tasks
                    if item["start"] <= start and item["end"] >= end
                ),
                None
            )

            if matching_task:
                entry = matching_task.copy()
                entry["start"] = start
                entry["end"] = end
                entry["type"] = "task"

            else:

                matching_block = next(
                    (
                        item for item in blocked
                        if item["start"] <= start and item["end"] >= end
                    ),
                    None
                )

                if matching_block:
                    entry = {
                        "type": matching_block["type"],
                        "task": matching_block["task"]
                    }

                else:
                    is_break = any(
                        item["end"] <= start
                        and start < item["end"] + timedelta(
                            minutes=break_minutes
                        )
                        for item in day_tasks
                    )

                    entry = {
                        "type": "break" if is_break else "break",
                        "task": "Break" if is_break else "Break"
                    }

                entry["start"] = start
                entry["end"] = end

            entry["date"] = current_date
            entry["minutes"] = int(
                (end - start).total_seconds() / 60
            )

            timetable.append(entry)

    # Merge adjacent entries of the same non-task type
    merged = []

    for entry in timetable:

        if (
            merged
            and entry["type"] != "task"
            and merged[-1]["type"] == entry["type"]
            and merged[-1]["date"] == entry["date"]
            and merged[-1]["end"] == entry["start"]
        ):
            merged[-1]["end"] = entry["end"]
            merged[-1]["minutes"] += entry["minutes"]

        else:
            merged.append(entry.copy())

    return merged


@app.route("/")
def home():
    return render_template("schedule.html")


@app.route("/create-schedule", methods=["GET"])
def create_schedule_page():

    return render_template("schedule.html")


@app.route("/schedule", methods=["POST"])
def schedule_setup():

    # Get information from task table

    task_names = request.form.getlist("task[]")
    categories = request.form.getlist("category[]")
    durations = request.form.getlist("duration[]")
    priorities = request.form.getlist("priority[]")
    deadlines = request.form.getlist("deadline[]")

    tasks = []

    # Turn each table row into a task

    for name, category, duration, priority, deadline in zip(
        task_names,
        categories,
        durations,
        priorities,
        deadlines
    ):

        # Ignore completely empty task rows
        if not name.strip():
            continue

        tasks.append({
            "task": name,
            "category": category,
            "duration": int(duration or 0),
            "priority": priority,
            "deadline": deadline
        })

    # Save tasks so page 2 can access them

    session["tasks"] = tasks

    print("TASKS SAVED:")
    print(tasks)

    return render_template("schedule_setup.html")

#generate the schedule


@app.route("/generate-schedule", methods=["POST"])
def create_schedule():

    # get tasks from page 1

    stored_tasks = session.get("tasks", [])

    tasks = []

    for task in stored_tasks:

        # Convert deadline string into datetime
        deadline = datetime.fromisoformat(
            task["deadline"]
        )

        tasks.append({
            "task": task["task"],
            "category": task["category"],
            "duration": task["duration"],
            "priority": task["priority"],
            "deadline": deadline
        })

    print("TASKS RECEIVED:")
    print(tasks)


    # day start/end

    start_time = datetime.strptime(
        request.form["start_time"],
        "%H:%M"
    ).time()

    end_time = datetime.strptime(
        request.form["end_time"],
        "%H:%M"
    ).time()

    # number of days

    number_of_days = int(
        request.form["days"]
    )

    # length of break

    break_minutes = int(
        request.form["break_length"]
    )

    # meal/break time

    meal_start_string = request.form.get(
        "meal_start"
    )

    meal_end_string = request.form.get(
        "meal_end"
    )

    meal_start = None
    meal_end = None

    if meal_start_string and meal_end_string:

        meal_start = datetime.strptime(
            meal_start_string,
            "%H:%M"
        ).time()

        meal_end = datetime.strptime(
            meal_end_string,
            "%H:%M"
        ).time()

    # Unavailable times

    unavailable_starts = request.form.getlist(
        "unavailable_start[]"
    )

    unavailable_ends = request.form.getlist(
        "unavailable_end[]"
    )

    unavailable_times = []

    for start, end in zip(
        unavailable_starts,
        unavailable_ends
    ):

        if start and end:

            unavailable_times.append(
                (
                    datetime.strptime(
                        start,
                        "%H:%M"
                    ).time(),

                    datetime.strptime(
                        end,
                        "%H:%M"
                    ).time()
                )
            )

    # generate schedule

    generated_schedule = generate_schedule(
        tasks=tasks,
        start_date=datetime.today().date(),
        start_time=start_time,
        end_time=end_time,
        number_of_days=number_of_days,
        break_minutes=break_minutes,
        meal_start=meal_start,
        meal_end=meal_end,
        unavailable_times=unavailable_times
    )

    generated_schedule = add_timetable_entries(
        schedule=generated_schedule,
        start_date=datetime.today().date(),
        start_time=start_time,
        end_time=end_time,
        number_of_days=number_of_days,
        break_minutes=break_minutes,
        meal_start=meal_start,
        meal_end=meal_end,
        unavailable_times=unavailable_times
    )

    print("GENERATED SCHEDULE:")
    print(generated_schedule)

    #calculate time remaining until deadline

    now = datetime.now()

    for item in generated_schedule:

        if item["type"] != "task":
            continue

        remaining = item["deadline"] - now

        total_minutes = int(remaining.total_seconds() // 60)

        if total_minutes < 0:
            item["time_remaining"] = "Overdue"

        else:
            days = total_minutes // 1440
            hours = (total_minutes % 1440) // 60
            minutes = total_minutes % 60

            item["time_remaining"] = (
                f"{days:02d}:{hours:02d}:{minutes:02d}"
            )

    # show result

    # Save the generated schedule
    session["generated_schedule"] = [
        {
            **item,
            "date": item["date"].isoformat(),
            "start": item["start"].isoformat(),
            "end": item["end"].isoformat(),
            "deadline": (
                item["deadline"].isoformat()
                if item.get("deadline")
                else None
            )
        }
        for item in generated_schedule
    ]

    session["number_of_days"] = number_of_days

    return redirect(url_for("show_schedule"))

# run website

@app.route("/your-schedule")
def show_schedule():

    stored_schedule = session.get("generated_schedule", [])

    schedule = []

    for item in stored_schedule:

        item = item.copy()

        # Convert stored dates and times back into Python objects
        item["date"] = datetime.fromisoformat(item["date"]).date()
        item["start"] = datetime.fromisoformat(item["start"])
        item["end"] = datetime.fromisoformat(item["end"])

        # Only actual tasks have deadlines
        if item.get("deadline"):
            item["deadline"] = datetime.fromisoformat(
                item["deadline"]
            )

        schedule.append(item)

    return render_template(
        "generated_schedule.html",
        schedule=schedule,
        number_of_days=session.get("number_of_days", 1)
    )



# study materials page

@app.route("/study-materials")
def study_materials():

    return render_template(
        "study_materials.html",
        result=None,
        mode=None
    )


# generate notes, flashcards, or quizzes

@app.route("/generate-study-material", methods=["POST"])
def generate_study():

    uploaded_file = request.files.get("pdf")
    mode = request.form.get("mode")

    if not uploaded_file or not uploaded_file.filename:
        return "Please upload a PDF.", 400

    if not uploaded_file.filename.lower().endswith(".pdf"):
        return "Only PDF files are supported.", 400

    if mode not in ("notes", "flashcards", "quiz"):
        return "Invalid generation option.", 400

    pdf_bytes = uploaded_file.read()

    # 15 MB maximum
    if len(pdf_bytes) > 15 * 1024 * 1024:
        return "PDF exceeds the 15 MB limit.", 400

    try:

        result = generate_study_material(
            pdf_bytes,
            mode
        )

        return render_template(
            "study_materials.html",
            result=result,
            mode=mode
        )
















    except Exception as error:

        import traceback

        print("\nSTUDY AI ERROR")

        traceback.print_exc()

        print("\n")

        return render_template(

            "study_materials.html",

            result=None,

            mode=None,

            error=f"Error type: {type(error).__name__}. Check PyCharm."

        ), 500





# to-do list page

@app.route("/todo-list")
def todo_list():

    return render_template("schedule_todolist.html")














if __name__ == "__main__":
    app.run(
        debug=True,
        port=5050
    )
