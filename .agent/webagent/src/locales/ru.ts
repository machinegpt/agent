import { TranslationDict } from "./types";

export const ru: TranslationDict = {
  header: {
    spec_interceptor: "MACHINE_GPT_LOOP // SPEC INTERCEPTOR",
    agent: "АГЕНТ",
    logs: "-ЛОГИ",
  },
  sidebar: {
    session_history: "История сессий",
    runs: "запусков",
    newer_live_available: "◉ Доступна новая сессия",
    switch_to_live: "Переключиться",
  },
  tabs: {
    summary: "Сводка",
    thoughts: "Мысли",
    files: "Файлы",
    console: "Логи",
    diffs: "Патчи",
  },
  session_info: {
    launched_at: "Запущено в: ",
    duration: "Длительность",
    workspace_files: "Файлы пространства",
  },
  cognitive_loop: {
    title: "Когнитивный цикл агента",
    status: "Статус",
  },
  thought_stream: {
    title: "Поток мыслей и монолог",
    desc: "Прямая трансляция когнитивных циклов и решений агента",
    search_placeholder: "Поиск мыслей...",
    category_all: "Все категории",
    phase_all: "Все фазы",
    no_thoughts: "Мыслей по вашему запросу не найдено",
  },
  file_explorer: {
    files_count: "файлов",
    no_files: "НЕТ ФАЙЛОВ В РАБОЧЕЙ ОБЛАСТИ",
    select_file_placeholder: "ВЫБЕРИТЕ ФАЙЛ ИЗ РАБОЧЕЙ ОБЛАСТИ .AGENT",
    copied: "Скопировано",
    copy_to_clipboard: "Копировать в буфер",
    copy_failed: "Ошибка копирования",
  },
  terminal: {
    terminal_io: "Журнал ввода/вывода терминала",
    rpc_ipc: "Перехватчик IPC JSON-RPC JINX",
    terminal_tab: "Вывод терминала",
    rpc_tab: "Поток IPC JSON-RPC",
    no_terminal: "[ ВЗАИМОДЕЙСТВИЙ С ТЕРМИНАЛОМ НЕ ЗАРЕГИСТРИРОВАНО ]",
    no_ipc: "[ СООБЩЕНИЙ IPC НЕ ЗАРЕГИСТРИРОВАНО ]",
    call_sent: "Вызов отправлен",
    reply_rcvd: "Ответ получен",
    parameters: "Параметры",
    response_result: "Результат ответа",
    response_error: "Исключение ответа",
  },
  diff_viewer: {
    no_diffs: "КОД-ДИФФЫ ИЛИ ИЗМЕНЕННЫЕ ФАЙЛЫ ЕЩЕ НЕ СГЕНЕРИРОВАНЫ",
  },
  run_summary: {
    multi_step_plan: "Прогресс многоэтапного плана агента",
    no_plan: "План еще не определен",
    elapsed_time: "Прошедшее время",
    subprocess_pid: "PID подпроцесса",
    host_node: "Хост-узел",
  },
  phases: {
    perceive: "Восприятие",
    analyze: "Анализ",
    plan: "Планирование",
    execute: "Выполнение",
    verify: "Проверка",
    commit: "Фиксация",
    completed: "Завершено",
    error: "Ошибка",
    idle: "Ожидание",
  },
  categories: {
    monologue: "Монолог",
    question: "Вопрос",
    decision: "Решение",
    check: "Валидация",
    system: "Система",
  },
  cognitive_loop_phases: {
    perceive: {
      label: "Восприятие",
      desc: "Сканирование кодовой базы и условий",
    },
    analyze: {
      label: "Анализ",
      desc: "Поиск первопричины и уязвимостей",
    },
    plan: {
      label: "План",
      desc: "Составление пошагового пути решения",
    },
    execute: {
      label: "Выполнение",
      desc: "Применение патчей файлов и запуск инструментов",
    },
    verify: {
      label: "Проверка",
      desc: "Запуск тестов, линтера и типов",
    },
    commit: {
      label: "Фиксация",
      desc: "Сохранение снимков и git-коммитов",
    },
    completed: {
      label: "Завершено",
      desc: "Подпроцесс остановлен, код зафиксирован",
    },
  },
};
