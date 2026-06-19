[English](README.md) | [Русский](README_RU.md) | [中文](README_ZH.md)

<p align="center">
  <img src="https://img.shields.io/badge/JINX-Enterprise_Agent_Runtime-000000?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyTDIgN2wxMCA1IDEwLTV6TTIgMTdsOCA0IDgtNE0yIDEybDggNCA4LTQiLz48L3N2Zz4=" alt="JINX Badge" />
  <img src="https://img.shields.io/badge/version-1.0.9--enterprise-blue?style=for-the-badge" alt="Version Badge" />
  <img src="https://img.shields.io/badge/architecture-Process_Isolated_IPC-red?style=for-the-badge" alt="Architecture Badge" />
  <img src="https://img.shields.io/badge/integration-Subprocess_Standard_Streams-brightgreen?style=for-the-badge" alt="Integration Badge" />
</p>

<h1 align="center">JINX — Спецификация Среды Выполнения Суверенного Корпоративного Агента</h1>

<p align="center">
  <strong>Техническая спецификация JINX, изолированного, сохраняющего состояние, управляемого протоколом когнитивного цикла, предназначенного для работы в качестве дочернего процесса внутри хост-среды разработки ПО.</strong>
</p>

---

## 1. Базовая архитектура и межпроцессное взаимодействие (IPC)

JINX представляет собой среду выполнения агента, спроектированную для запуска внутри хост-окружения (такого как IDE, консольный текстовый редактор или корпоративный оркестратор). Среда выполнения JINX функционирует без автономного сетевого доступа или встроенных интеграций с внешними сервисами; все запросы на вызов моделей, манипуляции с файлами и выполнение консольных команд делегируются хост-редактору через стандартный ввод (`stdin`) и стандартный вывод (`stdout`) с использованием структурированных пакетов обмена данными в формате JSON-RPC.

```text
┌─────────────────────────────────┐                 stdout (Пакеты JSON IPC)                 ┌─────────────────────────────────┐
│                                 │ ───────────────────────────────────────────────────────> │                                 │
│   Среда Выполнения JINX         │     "jinx_command": "llm_generate" / "bash_exec"         │   Хост-IDE / CLI-Редактор       │
│   (Стейт-машина и протокол)     │                                                          │   (Управляет API и запуском)    │
│   (Локальное состояние в YAML)  │ <─────────────────────────────────────────────────────── │                                 │
└─────────────────────────────────┘             stdin (Результат инструмента / ответ LLM)    └─────────────────────────────────┘
```

### Спецификация взаимодействия по протоколу JSON-RPC

При выполнении действия JINX выводит структурированный объект JSON в `stdout`, завершающийся символом новой строки. Хост-среда считывает этот объект из потока процесса, выполняет запрашиваемое действие и возвращает ответ в виде JSON-строки в `stdin` JINX, также завершающийся символом новой строки.

#### 1. Запрос генерации LLM (`llm_generate`)
JINX делегирует выполнение вызова LLM хосту.
* **Пакет, отправляемый в `stdout`**:
```json
{
  "jinx_command": "llm_generate",
  "params": {
    "system": "Системные инструкции, определяющие когнитивные границы.",
    "messages": [{"role": "user", "content": "Контекст конкретного раунда выполнения."}],
    "tools": [
      {
        "name": "bash_exec",
        "description": "Execute a bash or shell script in the environment.",
        "input_schema": {
          "type": "object",
          "properties": {
            "script": {"type": "string", "description": "The script to execute"}
          },
          "required": ["script"]
        }
      },
      {
        "name": "file_read",
        "description": "Read the contents of a file.",
        "input_schema": {
          "type": "object",
          "properties": {
            "path": {"type": "string", "description": "Path to the file"}
          },
          "required": ["path"]
        }
      },
      {
        "name": "file_write",
        "description": "Write or overwrite a file with new content.",
        "input_schema": {
          "type": "object",
          "properties": {
            "path": {"type": "string", "description": "Path to the file"},
            "content": {"type": "string", "description": "The full content to write"}
          },
          "required": ["path", "content"]
        }
      }
    ]
  }
}
```
* **Ожидаемый ответ хоста на `stdin`**:
```json
{
  "content": [
    {"type": "text", "text": "Анализ структуры кодовой базы."},
    {"type": "tool_use", "id": "call_123", "name": "bash_exec", "input": {"script": "pytest tests/test_core.py"}}
  ]
}
```

#### 2. Запуск команд консоли (`bash_exec`)
JINX запрашивает у хоста выполнение команды в консоли.
* **Пакет, отправляемый в `stdout`**:
```json
{
  "jinx_command": "bash_exec",
  "tool_use_id": "call_123",
  "params": {
    "script": "pytest tests/test_core.py"
  }
}
```
* **Ожидаемый ответ хоста на `stdin`**:
```json
{
  "output": "=== 1 passed in 0.05s ==="
}
```

#### 3. Операции с файлами (`file_read` и `file_write`)
JINX делегирует чтение и запись файлов хосту.
* **Пакет, отправляемый в `stdout` (чтение)**:
```json
{
  "jinx_command": "file_read",
  "tool_use_id": "call_124",
  "params": {
    "path": "src/core.py"
  }
}
```
* **Ожидаемый ответ хоста на `stdin` (чтение)**:
```json
{
  "content": "def run():\n    pass"
}
```

* **Пакет, отправляемый в `stdout` (запись)**:
```json
{
  "jinx_command": "file_write",
  "tool_use_id": "call_125",
  "params": {
    "path": "src/core.py",
    "content": "def run():\n    return True"
  }
}
```
* **Ожидаемый ответ хоста на `stdin` (запись)**:
```json
{
  "output": "Success"
}
```

---

## 2. Протокол выполнения когнитивного цикла

Работа JINX управляется итерационным циклом, выполняемым по четко разграниченным фазам. Параметры состояния сохраняются между итерациями в файле `JINX.yaml`.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {"darkMode": true, "background": "#0d1117", "primaryColor": "#21262d", "primaryTextColor": "#e6edf3", "primaryBorderColor": "#8b949e", "lineColor": "#8b949e", "textColor": "#e6edf3", "edgeLabelBackground": "#161b22", "mainBkg": "#21262d", "nodeBorder": "#8b949e", "nodeTextColor": "#e6edf3"}}}%%
graph LR
    classDef sub fill:#161b22,stroke:#30363d,stroke-dasharray: 3 3,color:#c9d1d9;
    classDef fail fill:#442326,stroke:#f85149,color:#ff7b72;
    classDef pass fill:#1f3b23,stroke:#56d364,color:#85e89d;

    subgraph P1["Фаза I: Сбор данных и скоупинг"]
        A["1. Анализ контекста и границ"]:::sub --> B["2. Запись границ в state.facts"]:::sub
    end
    style P1 fill:#0d1117,stroke:#30363d,color:#e6edf3

    subgraph P2["Фаза II: Генерация гипотез"]
        C["3. Регистрация истории сбоев"]:::sub --> D["4. Оценка альтернативных стратегий"]:::sub
    end
    style P2 fill:#0d1117,stroke:#30363d,color:#e6edf3

    subgraph P3["Фаза III: Разрушающее тестирование"]
        E["5. Запуск граничного тестирования"]:::sub --> F["6. Заполнение схемы требований"]:::sub
    end
    style P3 fill:#0d1117,stroke:#30363d,color:#e6edf3

    subgraph P4["Фаза IV: Оценка и выход"]
        G{"7. Проверка сходимости цикла"}:::sub
        G -->|Все пройдены| H["Успешный выход"]:::pass
        G -->|Сбои подходов >= 3| I["Триггер дедлока"]:::fail
        G -->|Раунды >= 40| J["Лимит раундов"]:::fail
    end
    style P4 fill:#0d1117,stroke:#30363d,color:#e6edf3

    B --> C
    D --> E
    F --> G
```

### Фазы выполнения

1. **Фаза I: Определение границ задачи и сбор данных**
   Прежде чем инициировать изменения файлов, JINX анализирует свойства рабочего пространства и фиксирует границы целевой задачи. Подтвержденный контекст записывается непосредственно в список `state.facts` конфигурационного манифеста `JINX.yaml`.

2. **Фаза II: Генерация гипотез и дивергенция**
   В случае неудачи предыдущего раунда JINX регистрирует причины сбоя в блоке `state.scores`. В последующих раундах JINX оценивает альтернативные технические стратегии. Повторение идентичных подходов без изменений заблокировано правилами протокола.

3. **Фаза III: Верификация граничных условий (Разрушающее тестирование / Breaker Test)**
   Для каждой технической стратегии должен выполняться этап граничного тестирования ("Breaker Test"). Реализация подлежит обязательной валидации на пограничных случаях, некорректных входных данных или пределах производительности. Критерии оценки структурированы в бинарной схеме (true/false) в блоке `state.scores[].requirements`.

4. **Фаза IV: Конвергенция и многокритериальный выход**
   После каждого раунда JINX обновляет метрики выполнения и проверяет условия выхода или дедлока:
   * **Условие выхода**: Проверяется, когда индекс раунда `round` больше или равен минимально заданному ограничению (`loop.min`), а параметр `exit_ready` установлен в значение `true`. Выход происходит, если последняя реализация удовлетворяет всем основным требованиям, и за последние 3 последовательных раунда не было получено более высокой оценки выполнения.
   * **Условие дедлока**: Активируется, если количество раундов превышает или равно значению `loop.min` и одно и то же требование падает на 3 независимых подходах. Также активируется при явном установлении флага `deadlock` в значение `true` внутри исполняемой среды.
   * **Жесткий лимит**: Общее количество итерационных раундов ограничено значением 40 (`HARD_CAP`), по достижении которого выполнение принудительно завершается для предотвращения избыточного расхода токенов.

### Диаграмма последовательности выполнения / Sequence Flow Diagram

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {"darkMode": true, "background": "#0d1117", "primaryColor": "#21262d", "primaryTextColor": "#e6edf3", "primaryBorderColor": "#8b949e", "lineColor": "#8b949e", "textColor": "#e6edf3", "edgeLabelBackground": "#161b22", "actorBkg": "#21262d", "actorBorder": "#8b949e", "actorTextColor": "#e6edf3", "actorLineColor": "#8b949e", "signalColor": "#8b949e", "signalTextColor": "#e6edf3", "noteBkgColor": "#373320", "noteBorderColor": "#d4a72c", "noteTextColor": "#f0e6c0", "labelBoxBkgColor": "#21262d", "labelBoxBorderColor": "#8b949e", "labelTextColor": "#e6edf3", "loopTextColor": "#e6edf3", "activationBkgColor": "#30363d", "activationBorderColor": "#8b949e"}}}%%
sequenceDiagram
    participant CLI as cli.py (main)
    participant Runner as runner.py (run)
    participant State as state.py
    participant Host as Host Editor (stdin/stdout)

    CLI->>Runner: run(task, min_override)
    Runner->>State: read_jinx()
    State-->>Runner: jinx dict
    Runner->>State: write_jinx(jinx) [init state]

    loop "Outer: rnd < HARD_CAP (40)"
        Runner->>State: read_jinx()
        State-->>Runner: current state

        loop "Inner: tool_depth < TOOL_DEPTH_CAP (20)"
            Runner->>Host: stdout JSON-RPC (llm_generate)
            Host-->>Runner: stdin content_blocks
            alt If tool_use detected
                loop For each tool_use
                    Runner->>Host: stdout JSON-RPC (tool call)
                    Host-->>Runner: stdin tool result
                end
            else No tool_use
                Note over Runner: Break Inner Loop
            end
        end

        Runner->>Runner: parse_state_block (last match)
        Runner->>State: merge_state + write_jinx
        alt exit_ready + check_exit
            Runner->>CLI: return (success)
        else deadlock detected or deadlock state
            Runner->>CLI: return (deadlock)
        else HARD_CAP exhausted
            Runner->>CLI: sys.exit(2)
        end
    end
```

---

## 3. Спецификация манифеста состояния (`JINX.yaml`)

Все когнитивные результаты, журналы ошибок, задачи и параметры цикла сериализуются в файл `JINX.yaml`, расположенный в изолированной рабочей папке `.agent`. Данная архитектура исключает хранение метаданных состояния в корне основного репозитория проекта.

```yaml
id: JINX
mode: editor-integrated

protocol:
  in:
    gap: infer; ask user only if wrong-guess cost = high or irreversible
    scope: reject surface ask; trace root cause, true scope, blast radius
    gate: before first line of code — write scope to state.facts; no code until written

  loop:
    min: 10
    try: new approach; differs from all prior
    test: use standard IDE tools (bash_exec, file_read, file_write)
    score: per-requirement pass/fail; rank vs all prior; write round + score to state.scores
    keep: top-scoring try persists
    branch: same requirement fails 3x same approach -> switch approach
    exit: iter >= 10 and top try passes all requirements and last 3 iters no higher score
    deadlock: iter >= 10 and 3 distinct approaches fail same requirement -> ask user, stop

state:
  task: "Текст описания задачи"
  facts:
    - "Fact 1: Workspace root verified"
    - "Fact 2: Configuration schema loaded"
  scores:
    - round: 1
      approach: "PyJWT RS256 token signing implementation"
      prior_failure: "None"
      requirements:
        compile: true
        unit_tests: false
      pass_count: 1
      all_pass: false
  debt: []
  open: []
  exit_ready: false
  deadlock: false
```

---

## 4. Инвентарь компонентов кодовой базы

Среда выполнения JINX состоит из следующих Python-компонентов, расположенных в директории `.agent/` (при этом основные пакетные модули находятся в `.agent/src/jinx/`):

* **`jinx.py`** (Входной скрипт запуска, расположен в `.agent/`):
  Служит единой точкой входа для выполнения. Настраивает пути импорта Python и делегирует обработку параметров командной строки парсеру.
* **`cli.py`** (Обработчик аргументов):
  Производит разбор входящих параметров с использованием библиотеки `argparse`. Собирает позиционный аргумент описания задачи и опциональный флаг переопределения минимального числа раундов `--min` перед вызовом основного оркестратора.
* **`runner.py`** (Оркестратор):
  Реализует логику конечного автомата. Содержит центральный цикл выполнения, управляет стандартными потоками для обмена сообщениями с хост-редактором, анализирует вывод модели для извлечения тегов состояния `<state>...</state>` и рассчитывает условия завершения работы или фиксации дедлока.
* **`state.py`** (Слой сериализации состояния):
  Выполняет операции файлового ввода-вывода для манифеста `JINX.yaml`. Использует схемы Pydantic (`ScoreEntry` и `StateBlock`) для валидации обновлений состояния и слияния изменений.
* **`tools.py`** (Вспомогательный модуль JSON-RPC):
  Задает схемы доступных инструментов (`bash_exec`, `file_read`, `file_write`), передаваемых в запросах LLM-генерации, и форматирует стандартизированный вывод в stdout.

---

## 5. Руководство по интеграции с хостом и реализации подпроцесса

Для интеграции JINX хост-редактор или оркестратор уровня предприятия должен запускать исполняемую команду JINX как дочерний процесс.

### Требования к запуску подпроцесса
* **Команда выполнения**: `python .agent/jinx.py "[TASK_DESCRIPTION]"`
* **Конфигурация процесса**: Настройте перенаправление потоков `stdout` и `stdin` в `subprocess.PIPE`. Активируйте текстовый режим (`text=True`) и обеспечьте автоматический сброс буфера вывода (`flush`).
* **Логика цикла**: Считывайте каждую строку из `stdout` как объект JSON, маршрутизируйте вызов в зависимости от значения поля `jinx_command`, выполняйте соответствующее системное действие и возвращайте результат в `stdin` в виде JSON-строки в одну строку.

### Пример интеграции на языке Python

Приведенный ниже сценарий демонстрирует практическую реализацию хост-стороны протокола IPC:

```python
import subprocess
import json

def execute_jinx(task_description: str):
    # Запуск JINX как дочернего процесса
    process = subprocess.Popen(
        ["python", ".agent/jinx.py", task_description],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        text=True
    )

    try:
        # Построчное чтение вывода из дочернего процесса JINX
        for line in process.stdout:
            payload = json.loads(line.strip())
            command = payload.get("jinx_command")
            tool_use_id = payload.get("tool_use_id")
            params = payload.get("params", {})

            if command == "llm_generate":
                # Реализация корпоративной логики генерации LLM
                # ...
                ai_output = [
                    {"type": "text", "text": "Текстовый шаг генерации."},
                    {"type": "tool_use", "id": "call_01", "name": "bash_exec", "input": {"script": "pytest"}}
                ]
                # Отправка JSON-ответа обратно в stdin JINX
                process.stdin.write(json.dumps({"content": ai_output}) + "\n")
                process.stdin.flush()

            elif command == "bash_exec":
                # Запуск команды в окружении хоста
                script = params.get("script")
                # ...
                execution_result = "Test suite passed"
                # Отправка JSON-ответа обратно в stdin JINX
                process.stdin.write(json.dumps({"output": execution_result}) + "\n")
                process.stdin.flush()

            elif command == "file_read":
                # Чтение локального файла рабочей области
                filepath = params.get("path")
                # ...
                file_content = "File content mock"
                process.stdin.write(json.dumps({"content": file_content}) + "\n")
                process.stdin.flush()

            elif command == "file_write":
                # Запись в локальный файл рабочей области
                filepath = params.get("path")
                content = params.get("content")
                # ...
                process.stdin.write(json.dumps({"output": "Success"}) + "\n")
                process.stdin.flush()

    except Exception as e:
        process.kill()
        raise e

    process.wait()
    return process.returncode

if __name__ == "__main__":
    exit_code = execute_jinx("Implement corporate schema update")
    print(f"Код завершения процесса JINX: {exit_code}")
```

---

## 6. Постинтеграционный рабочий процесс разработчика

После успешного запуска JINX и настройки управления IPC-подключением со стороны хост-редактора, взаимодействие разработчика с прошивкой строится на основе модели аудита и оперативного вмешательства.

### Диагностика в реальном времени
Во время работы JINX разработчику не требуется вручную обрабатывать стандартные потоки ввода-вывода (эти задачи полностью выполняет фоновый модуль интеграции IDE). Вместо этого разработчик может контролировать ход выполнения по следующим каналам:
1. **Аудит манифеста состояния**:
   Откройте `.agent/JINX.yaml` в редакторе. Этот файл автоматически обновляется в реальном времени по окончании каждого раунда. Блок `state` выступает в качестве интерактивного дашборда:
   * **`facts`**: Отслеживает все выявленные факты и свойства рабочей области, принятые агентом.
   * **`scores`**: Регистрирует метрики и результаты выполнения каждого раунда с детализацией по требованиям.
   * **`debt`**: Перечисляет зафиксированные компромиссы или временные решения.
2. **Просмотр логов генерации**:
   Интеграционный модуль IDE перехватывает промежуточные мыслительные блоки LLM (`{"type": "text"}`) и транслирует их в нативную вкладку чата или консоли, что позволяет видеть текущую когнитивную задачу агента.

### Обработка пауз и разрешение дедлоков
Архитектура JINX предусматривает принудительную остановку цикла при достижении критических лимитов протокола для запроса вмешательства человека:
* **Условие дедлока**:
  Если одно и то же требование падает на 3 независимых подходах, флаг состояния переходит в значение `deadlock: true`, и дочерний процесс завершает работу с ошибкой либо приостанавливается.
* **Рабочий процесс ручной коррекции**:
  1. Разработчик открывает файл `.agent/JINX.yaml` для локализации упавшего требования и истории попыток.
  2. Разработчик устраняет блокирующую проблему в коде проекта вручную или корректирует параметры окружения (например, исправляет конфигурации БД или тестовые фикстуры).
  3. При необходимости разработчик может вручную изменить свойства `state` в `JINX.yaml` (например, скорректировать факты или список открытых задач).
  4. Разработчик повторно запускает сессию JINX из CLI через команду хоста. JINX считывает текущий манифест `JINX.yaml`, идентифицирует историю прошлых раундов и продолжает когнитивный цикл с учетом обновленного контекста.

### Верификация и фиксация изменений
Когда когнитивный цикл успешно завершает работу по критериям выхода, процесс JINX завершается с кодом `0`.
1. **Анализ диффов**: Разработчик проверяет внесенные изменения в файлах проекта.
2. **Фиксация кода**: Разработчик выполняет коммит готовых файлов. Метаданные состояния в `.agent/JINX.yaml` остаются в изолированной директории рабочего пространства и служат контекстной базой для последующих запусков.
