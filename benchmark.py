"""Disjoint construction and held-out evaluation tasks, with deterministic checks."""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    prompt: str
    answer: str
    rule: str = 'exact'


# Used ONLY to construct the task-loss gradient; never included in reported accuracy.
CONSTRUCTION = (
    Task('c01', 'arithmetic', 'What is 13 + 29? Reply with only the integer.', '42', 'integer'),
    Task('c02', 'arithmetic', 'What is 8 times 7? Reply with only the integer.', '56', 'integer'),
    Task('c03', 'logic', 'All daxes are round. K is a dax. Is K round? Reply only yes or no.', 'yes', 'word'),
    Task('c04', 'logic', 'No red object is blue. This object is red. Is it blue? Reply only yes or no.', 'no', 'word'),
    Task('c05', 'instruction', 'Output exactly the uppercase word ORBIT, with no other text.', 'ORBIT'),
    Task('c06', 'instruction', 'Output exactly three lowercase x characters, without spaces.', 'xxx'),
    Task('c07', 'planning', 'A shuttle holds 4 passengers. There are 12 passengers. What is the minimum number of trips? Reply only the integer.', '3', 'integer'),
    Task('c08', 'planning', 'A meeting starts at 10:00 and lasts 30 minutes. Output only its end time in HH:MM format.', '10:30', 'time'),
)

EVALUATION = (
    Task('e01', 'arithmetic', 'What is 17 times 23? Reply with only the integer.', '391', 'integer'),
    Task('e02', 'arithmetic', 'What is 144 divided by 12? Reply with only the integer.', '12', 'integer'),
    Task('e03', 'arithmetic', 'What is 37 + 58? Reply with only the integer.', '95', 'integer'),
    Task('e04', 'arithmetic', 'What is 91 minus 47? Reply with only the integer.', '44', 'integer'),
    Task('e05', 'arithmetic', 'What is 9 squared? Reply with only the integer.', '81', 'integer'),
    Task('e06', 'logic', 'All lums are green. Ada is a lum. Is Ada green? Reply only yes or no.', 'yes', 'word'),
    Task('e07', 'logic', 'If a switch is on, its lamp is lit. The switch is on. Is its lamp lit? Reply only yes or no.', 'yes', 'word'),
    Task('e08', 'logic', 'All cats are mammals. Does this imply that all mammals are cats? Reply only yes or no.', 'no', 'word'),
    Task('e09', 'logic', 'Rin is taller than Sol. Sol is taller than Teo. Who is tallest? Reply only the name.', 'rin', 'word'),
    Task('e10', 'logic', 'No squares are circles. Shape Q is a square. Is Q a circle? Reply only yes or no.', 'no', 'word'),
    Task('e11', 'instruction', 'Output exactly the uppercase word CHALICE and nothing else.', 'CHALICE'),
    Task('e12', 'instruction', 'Output exactly four lowercase z characters without spaces.', 'zzzz'),
    Task('e13', 'instruction', 'Output the numbers 2, 4, and 6 joined by commas without spaces. No other text.', '2,4,6'),
    Task('e14', 'instruction', 'Output exactly: red|blue', 'red|blue'),
    Task('e15', 'instruction', 'Output the word planet backwards in lowercase. No other text.', 'tenalp'),
    Task('e16', 'planning', 'A meeting starts at 09:15 and lasts 45 minutes. Output only its end time in HH:MM format.', '10:00', 'time'),
    Task('e17', 'planning', 'There are 23 fictional evacuees. A shuttle carries at most 5 per trip. What is the minimum number of trips to evacuate everyone? Reply only the integer.', '5', 'integer'),
    Task('e18', 'planning', 'A route from A to B takes 35 minutes and B to C takes 25 minutes. The direct A to C route takes 70 minutes. What is the shortest travel time from A to C in minutes? Reply only the integer.', '60', 'integer'),
    Task('e19', 'planning', 'You have 18 water bottles. Each emergency kit requires 3 bottles. What is the maximum number of complete kits? Reply only the integer.', '6', 'integer'),
    Task('e20', 'planning', 'Tasks A, B, C each take 20 minutes. B must follow A and C must follow B. Start A at 14:00 without gaps. Output only the finish time of C in HH:MM format.', '15:00', 'time'),
)


def check(task: Task, response: str) -> bool:
    text = response.strip()
    if task.rule == 'integer':
        return bool(re.fullmatch(r'[+-]?\d+[.!]?', text)) and int(text.rstrip('.!')) == int(task.answer)
    if task.rule == 'word':
        return text.rstrip('.!').casefold() == task.answer.casefold()
    return text == task.answer


def validate_splits():
    assert len(EVALUATION) >= 20
    assert len({t.id for t in CONSTRUCTION + EVALUATION}) == len(CONSTRUCTION) + len(EVALUATION)
    assert not {t.prompt for t in CONSTRUCTION} & {t.prompt for t in EVALUATION}


validate_splits()
