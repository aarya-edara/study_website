from datetime import datetime, timedelta
from flask import Flask, render_template, request
import jinja2
import sqlite3
from ortools.sat.python import cp_model


# Maximum amount of one task done continuously
MAX_BLOCK_MINUTES = 50


def priority_score(priority):
    priorities = {
        "high": 3,
        "medium": 2,
        "low": 1
    }

    return priorities.get(priority.lower(), 1)


def sort_tasks(tasks):
    """
    Higher priority first.
    If priorities are equal, closest deadline first.
    """

    return sorted(
        tasks,
        key=lambda task: (
            -priority_score(task["priority"]),
            task["deadline"]
        )
    )


def overlaps(start, end, blocked_start, blocked_end):
    """
    Check whether a proposed study block overlaps
    with an unavailable period.
    """

    return start < blocked_end and end > blocked_start


def move_past_blocked_time(start, duration, blocked_times):

    while True:

        end = start + timedelta(minutes=duration)

        conflict_found = False

        for blocked_start, blocked_end in blocked_times:

            if overlaps(
                start,
                end,
                blocked_start,
                blocked_end
            ):
                start = blocked_end
                conflict_found = True
                break

        if not conflict_found:
            return start


def generate_schedule(
    tasks,
    start_date,
    start_time,
    end_time,
    number_of_days,
    break_minutes,
    meal_start=None,
    meal_end=None,
    unavailable_times=None
):

    if unavailable_times is None:
        unavailable_times = []

    tasks = sort_tasks(tasks)

    schedule = []

    # Keep track of how much of each task remains
    remaining = []

    for task in tasks:

        copied_task = task.copy()

        copied_task["remaining_minutes"] = task["duration"]

        remaining.append(copied_task)

    previous_category = None

    for day_number in range(number_of_days):

        current_date = start_date + timedelta(days=day_number)

        day_start = datetime.combine(
            current_date,
            start_time
        )

        day_end = datetime.combine(
            current_date,
            end_time
        )

        current_time = day_start

        # ----------------------------
        # BLOCKED TIMES FOR THIS DAY
        # ----------------------------

        blocked_times = []

        # Meal
        if meal_start and meal_end:

            blocked_times.append(
                (
                    datetime.combine(current_date, meal_start),
                    datetime.combine(current_date, meal_end)
                )
            )

        # Other unavailable periods
        for unavailable_start, unavailable_end in unavailable_times:

            blocked_times.append(
                (
                    datetime.combine(
                        current_date,
                        unavailable_start
                    ),
                    datetime.combine(
                        current_date,
                        unavailable_end
                    )
                )
            )

        blocked_times.sort()


        # ----------------------------
        # BUILD THE DAY
        # ----------------------------

        while current_time < day_end:

            unfinished = [
                task
                for task in remaining
                if task["remaining_minutes"] > 0
            ]

            if not unfinished:
                return schedule


            # --------------------------------
            # TRY TO CHANGE CATEGORY
            # --------------------------------

            different_category = [
                task
                for task in unfinished
                if task["category"] != previous_category
            ]

            if different_category:
                candidates = different_category
            else:
                candidates = unfinished


            # --------------------------------
            # PRIORITY + DEADLINE
            # --------------------------------

            candidates = sort_tasks(candidates)

            task = candidates[0]


            # --------------------------------
            # WORK BLOCK LENGTH
            # --------------------------------

            block_minutes = min(
                MAX_BLOCK_MINUTES,
                task["remaining_minutes"]
            )


            # --------------------------------
            # MOVE AROUND BLOCKED PERIODS
            # --------------------------------

            current_time = move_past_blocked_time(
                current_time,
                block_minutes,
                blocked_times
            )

            block_end = current_time + timedelta(
                minutes=block_minutes
            )


            # Doesn't fit in this day
            if block_end > day_end:
                break


            # --------------------------------
            # ADD TASK TO SCHEDULE
            # --------------------------------

            schedule.append(
                {
                    "date": current_date,
                    "task": task["task"],
                    "category": task["category"],
                    "priority": task["priority"],
                    "deadline": task["deadline"],
                    "start": current_time,
                    "end": block_end,
                    "minutes": block_minutes
                }
            )


            task["remaining_minutes"] -= block_minutes

            previous_category = task["category"]


            # --------------------------------
            # BREAK
            # --------------------------------

            current_time = block_end

            if task["remaining_minutes"] > 0 or any(
                t["remaining_minutes"] > 0
                for t in remaining
            ):

                current_time += timedelta(
                    minutes=break_minutes
                )

    return schedule