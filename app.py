import streamlit as st
import pandas as pd
import json
import os
import datetime
import networkx as nx
import math
import numpy as np
import plotly.graph_objects as go
import colorsys

# Настройка страницы с адаптивным макетом
st.set_page_config(page_title="Фамильное древо", layout="wide", initial_sidebar_state="collapsed")

# --- Функции визуализации в начале файла ---

def calculate_relation_levels(members, relationships, central_person_id):
    """
    Вычисляет уровни родства всех членов семьи относительно центрального узла
    
    Returns:
        dict: Словарь {id: уровень}, где уровень:
            0 - центральный человек
            1 - прямая семья (родители, дети, супруг/а)
            2 - близкие родственники (бабушки/дедушки, братья/сестры)
            3 - дальние родственники (дяди/тети, двоюродные)
    """
    levels = {central_person_id: 0}  # Центральный узел
    
    # Строим граф для анализа связей
    G = build_family_graph(members, relationships)
    
    # Находим родителей, детей и супруга
    parents = list(G.predecessors(central_person_id))
    children = list(G.successors(central_person_id))
    
    # Находим супруга (общие дети)
    spouse_ids = []
    for child_id in children:
        child_parents = list(G.predecessors(child_id))
        for parent_id in child_parents:
            if parent_id != central_person_id:
                spouse_ids.append(parent_id)
    
    # Прямая семья (уровень 1)
    for member_id in parents + children + spouse_ids:
        levels[member_id] = 1
    
    # Находим братьев/сестер (общие родители)
    siblings = []
    for parent_id in parents:
        for child_id in G.successors(parent_id):
            if child_id != central_person_id and child_id not in siblings:
                siblings.append(child_id)
    
    # Находим бабушек/дедушек (родители родителей)
    grandparents = []
    for parent_id in parents:
        for gp_id in G.predecessors(parent_id):
            if gp_id not in grandparents:
                grandparents.append(gp_id)
    
    # Близкие родственники (уровень 2)
    for member_id in siblings + grandparents:
        levels[member_id] = 2
    
    # Дяди/тети (братья/сестры родителей)
    uncles_aunts = []
    for parent_id in parents:
        parent_siblings = []
        parent_parents = list(G.predecessors(parent_id))
        for gp_id in parent_parents:
            for child_id in G.successors(gp_id):
                if child_id != parent_id and child_id not in uncles_aunts:
                    uncles_aunts.append(child_id)
    
    # Племянники (дети братьев/сестер)
    niblings = []
    for sibling_id in siblings:
        for child_id in G.successors(sibling_id):
            if child_id not in niblings:
                niblings.append(child_id)
    
    # Двоюродные (дети дядь/теть)
    cousins = []
    for ua_id in uncles_aunts:
        for child_id in G.successors(ua_id):
            if child_id not in cousins:
                cousins.append(child_id)
    
    # Дальние родственники (уровень 3)
    for member_id in uncles_aunts + niblings + cousins:
        levels[member_id] = 3
    
    return levels

def get_relation_to_person(members, relationships, central_id, person_id):
    """
    Определяет отношение человека к центральному узлу
    """
    if central_id == person_id:
        return "Это я"
    
    # Строим граф для анализа отношений
    G = nx.DiGraph()
    for member in members:
        G.add_node(member["id"], **member)
    for rel in relationships:
        G.add_edge(rel["parent_id"], rel["child_id"])
    
    # Находим прямых родителей центрального узла
    parents = list(G.predecessors(central_id))
    
    # Находим прямых детей центрального узла
    children = list(G.successors(central_id))
    
    # Находим братьев/сестер центрального узла (имеют тех же родителей)
    siblings = []
    for parent_id in parents:
        for child_id in G.successors(parent_id):
            if child_id != central_id and child_id not in siblings:
                siblings.append(child_id)
    
    # Проверяем родительские связи
    if person_id in parents:
        person = next(m for m in members if m["id"] == person_id)
        return "Отец" if person["gender"] == "Мужской" else "Мать"
    
    # Проверяем супружеские связи
    marriage_pairs = find_marriage_pairs(relationships)
    for pair in marriage_pairs:
        if central_id in pair and person_id in pair:
            person = next(m for m in members if m["id"] == person_id)
            return "Муж" if person["gender"] == "Мужской" else "Жена"
    
    # Проверяем, является ли человек ребенком
    if person_id in children:
        person = next(m for m in members if m["id"] == person_id)
        return "Сын" if person["gender"] == "Мужской" else "Дочь"
    
    # Проверяем, является ли человек братом/сестрой
    if person_id in siblings:
        person = next(m for m in members if m["id"] == person_id)
        return "Брат" if person["gender"] == "Мужской" else "Сестра"
    
    # Проверяем бабушек/дедушек (родители родителей)
    for parent_id in parents:
        grandparents = list(G.predecessors(parent_id))
        if person_id in grandparents:
            person = next(m for m in members if m["id"] == person_id)
            return "Дедушка" if person["gender"] == "Мужской" else "Бабушка"
    
    # Проверяем дядей/теть (братья/сестры родителей)
    uncles_aunts = []
    for parent_id in parents:
        parent_parents = list(G.predecessors(parent_id))
        for grandparent in parent_parents:
            for uncle_aunt in G.successors(grandparent):
                if uncle_aunt != parent_id and uncle_aunt not in uncles_aunts:
                    uncles_aunts.append(uncle_aunt)
    
    if person_id in uncles_aunts:
        person = next(m for m in members if m["id"] == person_id)
        return "Дядя" if person["gender"] == "Мужской" else "Тетя"
    
    # Проверяем двоюродных братьев/сестер (дети дядей/теть)
    cousins = []
    for uncle_aunt in uncles_aunts:
        for cousin in G.successors(uncle_aunt):
            if cousin not in cousins:
                cousins.append(cousin)
    
    if person_id in cousins:
        person = next(m for m in members if m["id"] == person_id)
        return "Двоюродный брат" if person["gender"] == "Мужской" else "Двоюродная сестра"
    
    # Проверяем племянников (дети братьев/сестер)
    niblings = []
    for sibling in siblings:
        for child in G.successors(sibling):
            if child not in niblings:
                niblings.append(child)
    
    if person_id in niblings:
        person = next(m for m in members if m["id"] == person_id)
        return "Племянник" if person["gender"] == "Мужской" else "Племянница"
    
    # По умолчанию, если отношение не определено
    return "Родственник"

def find_marriage_pairs(relationships):
    """
    Находит супружеские пары на основе общих детей
    """
    # Словарь для группировки родителей по детям
    parents_by_child = {}
    
    for rel in relationships:
        child_id = rel["child_id"]
        parent_id = rel["parent_id"]
        
        if child_id not in parents_by_child:
            parents_by_child[child_id] = []
        
        parents_by_child[child_id].append(parent_id)
    
    # Находим пары родителей, у которых есть общие дети
    marriage_pairs = set()
    for child_id, parents in parents_by_child.items():
        if len(parents) >= 2:
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    pair = tuple(sorted([parents[i], parents[j]]))
                    marriage_pairs.add(pair)
    
    return list(marriage_pairs)

def get_relation_group(relation):
    """
    Группирует типы отношений для размещения на концентрических кругах
    """
    parent_relations = ["Отец", "Мать"]
    child_relations = ["Сын", "Дочь"]
    spouse_relations = ["Муж", "Жена"]
    grandparent_relations = ["Дедушка", "Бабушка"]
    sibling_relations = ["Брат", "Сестра"]
    uncle_aunt_relations = ["Дядя", "Тетя"]
    cousin_relations = ["Двоюродный брат", "Двоюродная сестра"]
    nibling_relations = ["Племянник", "Племянница"]
    
    if relation in parent_relations:
        return "parents"
    elif relation in child_relations:
        return "children"
    elif relation in spouse_relations:
        return "spouse"
    elif relation in grandparent_relations:
        return "grandparents"
    elif relation in sibling_relations:
        return "siblings"
    elif relation in uncle_aunt_relations:
        return "uncles_aunts"
    elif relation in cousin_relations:
        return "cousins"
    elif relation in nibling_relations:
        return "niblings"
    else:
        return "other"

def get_node_color(gender, level, color_scheme="standard"):
    """
    Определяет цвет узла в зависимости от пола, уровня родства и цветовой схемы
    """
    if color_scheme == "standard":
        if gender == "Мужской":
            # Оттенки синего для мужчин
            colors = ["#0047AB", "#1E88E5", "#42A5F5", "#64B5F6", "#90CAF9"]
        else:
            # Оттенки розового для женщин
            colors = ["#FF1493", "#FF69B4", "#FF80AB", "#F8BBD0", "#FCE4EC"]
    
    elif color_scheme == "contrast":
        if gender == "Мужской":
            # Контрастные синие
            colors = ["#003366", "#0066CC", "#3399FF", "#66CCFF", "#99FFFF"]
        else:
            # Контрастные красные
            colors = ["#990000", "#CC0000", "#FF0000", "#FF6666", "#FFCCCC"]
    
    elif color_scheme == "monochrome":
        # Монохромная схема, разные оттенки серого
        if gender == "Мужской":
            colors = ["#222222", "#444444", "#666666", "#888888", "#AAAAAA"]
        else:
            colors = ["#333333", "#555555", "#777777", "#999999", "#BBBBBB"]
    
    else:  # Стандартная схема по умолчанию
        if gender == "Мужской":
            colors = ["#0047AB", "#1E88E5", "#42A5F5", "#64B5F6", "#90CAF9"]
        else:
            colors = ["#FF1493", "#FF69B4", "#FF80AB", "#F8BBD0", "#FCE4EC"]
    
    idx = min(level, len(colors)-1)
    return colors[idx]

def create_concentric_family_tree(members, relationships, central_person_id=3, show_names=True, show_relations=True, color_scheme="standard"):
    """
    Создает концентрическую визуализацию семейного древа с заданным центральным узлом.
    
    Args:
        members: Список словарей с информацией о членах семьи
        relationships: Список словарей с информацией о родственных связях
        central_person_id: ID члена семьи, который будет в центре (по умолчанию Георгий Богданов, ID=3)
        show_names: Показывать ли полные имена
        show_relations: Показывать ли родственные связи
        color_scheme: Цветовая схема ("standard", "contrast", "monochrome")
        
    Returns:
        fig: Объект plotly Figure с визуализацией
    """
    import plotly.io as pio
    
    # Определяем, запущено ли приложение на мобильном устройстве
    is_mobile = False
    try:
        if 'is_mobile' in st.session_state:
            is_mobile = st.session_state.is_mobile
    except ImportError:
        pass
    
    # Вычисляем степень родства для каждого члена семьи относительно центрального узла
    relation_levels = calculate_relation_levels(members, relationships, central_person_id)
    
    # Получаем центрального человека
    central_person = next((m for m in members if m["id"] == central_person_id), None)
    if not central_person:
        return None
    
    # Создаем граф для анализа
    G = build_family_graph(members, relationships)
    
    # Подготавливаем данные для визуализации
    nodes_by_level = {}
    labels = {}
    node_colors = {}
    
    for member in members:
        member_id = member["id"]
        level = relation_levels.get(member_id, 4)  # Если уровень не определен, помещаем на 4й круг
        
        if level not in nodes_by_level:
            nodes_by_level[level] = []
        
        nodes_by_level[level].append(member_id)
        
        # Определяем метку с именем и родством
        relation = get_relation_to_person(members, relationships, central_person_id, member_id)
        
        # Для мобильного отображения - компактная версия имени
        name_display = member['name']
        if is_mobile:
            # На мобильных устройствах укорачиваем длинные имена
            name_parts = member['name'].split()
            if len(name_parts) > 2:
                if len(name_parts[0]) > 8 or len(name_parts[1]) > 8:
                    # Если имя или фамилия длинные, показываем только первую букву отчества
                    name_display = f"{name_parts[0]} {name_parts[1][0]}. {name_parts[-1]}"
        
        if member_id == central_person_id:
            if show_names:
                labels[member_id] = f"{name_display}<br>(Центр древа)"
            else:
                labels[member_id] = f"(Центр древа)"
        else:
            if show_names and show_relations:
                labels[member_id] = f"{name_display}<br>({relation})"
            elif show_names:
                labels[member_id] = f"{name_display}"
            elif show_relations:
                labels[member_id] = f"({relation})"
            else:
                labels[member_id] = f"#{member_id}"
        
        # Определяем цвет узла в зависимости от пола и выбранной цветовой схемы
        node_colors[member_id] = get_node_color(member["gender"], level, color_scheme)
    
    # Создаем фигуру plotly
    fig = go.Figure()
    
    # Расставляем узлы по концентрическим кругам
    max_level = max(nodes_by_level.keys()) if nodes_by_level else 4
    radius_step = 1.0 / max_level if max_level > 0 else 1.0
    
    node_positions = {}
    
    # Размещаем центральный узел в центре
    node_positions[central_person_id] = (0, 0)
    
    # Размещаем остальные узлы по концентрическим кругам
    for level, node_ids in nodes_by_level.items():
        if level == 0:  # Центральный узел уже размещен
            continue
        
        radius = level * radius_step
        node_count = len(node_ids)
        
        # Определяем начальный угол в зависимости от типа отношения
        start_angles = {
            1: {  # Прямая семья (родители - верхний полукруг, дети - нижний)
                "parents": 45,      # Родители в верхней части
                "children": 225,    # Дети в нижней части
                "spouse": 135       # Супруги справа
            },
            2: {  # Расширенная семья
                "grandparents": 30,  # Бабушки/дедушки вверху
                "siblings": 100,     # Братья/сестры справа
                "niblings": 260      # Племянники внизу-справа
            }
        }
        
        # Группируем узлы по типу отношения
        grouped_nodes = {}
        for node_id in node_ids:
            relation = get_relation_to_person(members, relationships, central_person_id, node_id)
            relation_type = get_relation_group(relation)
            if relation_type not in grouped_nodes:
                grouped_nodes[relation_type] = []
            grouped_nodes[relation_type].append(node_id)
        
        # Размещаем узлы по группам
        for group, group_nodes in grouped_nodes.items():
            if level in start_angles and group in start_angles[level]:
                start_angle = start_angles[level][group]
            else:
                start_angle = 0
            
            angle_step = 360 / max(20, len(node_ids))  # Минимум 20 для предотвращения перекрытий
            
            for i, node_id in enumerate(group_nodes):
                angle = (start_angle + i * angle_step) % 360
                x = radius * math.cos(math.radians(angle))
                y = radius * math.sin(math.radians(angle))
                node_positions[node_id] = (x, y)
    
    # Добавляем узлы (члены семьи) с метками
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    node_ids = []
    
    for member in members:
        member_id = member["id"]
        if member_id in node_positions:
            x, y = node_positions[member_id]
            node_x.append(x)
            node_y.append(y)
            node_text.append(labels[member_id])
            node_color.append(node_colors[member_id])
            
            # Размер узлов адаптируется для мобильных устройств
            if is_mobile:
                # На мобильных делаем узлы больше для удобства тач-интерфейса
                node_size.append(50 if member_id == central_person_id else 40)
            else:
                # На десктопах стандартный размер
                node_size.append(40 if member_id == central_person_id else 30)
                
            node_ids.append(member_id)
    
    # Добавляем концентрические круги для контекста
    for level in range(1, max_level+1):
        radius = level * radius_step
        circle_x = []
        circle_y = []
        
        for angle in range(0, 361, 1):
            circle_x.append(radius * math.cos(math.radians(angle)))
            circle_y.append(radius * math.sin(math.radians(angle)))
        
        circle_trace = go.Scatter(
            x=circle_x,
            y=circle_y,
            mode='lines',
            line=dict(width=0.5, color='lightgrey'),
            hoverinfo='none'
        )
        fig.add_trace(circle_trace)
    
    # Теперь добавляем связи между узлами, чтобы они были под узлами
    # Родительские связи (сплошные линии)
    parent_edge_x = []
    parent_edge_y = []
    
    for rel in relationships:
        parent_id = rel["parent_id"]
        child_id = rel["child_id"]
        
        if parent_id in node_positions and child_id in node_positions:
            x0, y0 = node_positions[parent_id]
            x1, y1 = node_positions[child_id]
            
            parent_edge_x.append(x0)
            parent_edge_x.append(x1)
            parent_edge_x.append(None)
            parent_edge_y.append(y0)
            parent_edge_y.append(y1)
            parent_edge_y.append(None)
    
    # Супружеские связи (пунктирные линии)
    marriage_edge_x = []
    marriage_edge_y = []
    
    marriage_pairs = find_marriage_pairs(relationships)
    for pair in marriage_pairs:
        id1, id2 = pair
        if id1 in node_positions and id2 in node_positions:
            x0, y0 = node_positions[id1]
            x1, y1 = node_positions[id2]
            
            marriage_edge_x.append(x0)
            marriage_edge_x.append(x1)
            marriage_edge_x.append(None)
            marriage_edge_y.append(y0)
            marriage_edge_y.append(y1)
            marriage_edge_y.append(None)
    
    # Рисуем родительские связи (сплошные линии)
    parent_child_edges = go.Scatter(
        x=parent_edge_x,
        y=parent_edge_y,
        mode='lines',
        line=dict(width=1, color='#888'),
        hoverinfo='none'
    )
    fig.add_trace(parent_child_edges)
    
    # Рисуем супружеские связи (пунктирные линии)
    marriage_edges = go.Scatter(
        x=marriage_edge_x,
        y=marriage_edge_y,
        mode='lines',
        line=dict(width=1, color='#FF6666', dash='dash'),
        hoverinfo='none'
    )
    fig.add_trace(marriage_edges)
    
    # Адаптивный размер шрифта и формат узлов в зависимости от устройства
    text_size = 10
    if is_mobile:
        text_size = 8  # Уменьшаем размер текста на мобильных
    
    # Добавляем узлы на график поверх линий
    nodes_trace = go.Scatter(
        x=node_x, 
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='DarkSlateGrey')
        ),
        text=[f"{i}" for i in node_ids],  # Короткий текст внутри узла
        hovertext=node_text,  # Полный текст для всплывающей подсказки
        hoverinfo='text',
        textposition="middle center",
        textfont=dict(size=text_size)
    )
    
    fig.add_trace(nodes_trace)
    
    # Настройка макета графика
    title = f"Фамильное древо - центр: {central_person['name']}"
    
    # Адаптивный макет в зависимости от устройства
    if is_mobile:
        # Более компактный макет для мобильных с минимальными полями
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=10, l=5, r=5, t=30),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white",
            dragmode="pan",  # Для мобильных лучше использовать режим панорамирования по умолчанию
            height=450,      # Меньшая высота графика
        )
    else:
        # Стандартный макет для десктопов
        fig.update_layout(
            title=title,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700,
            template="plotly_white"
        )
    
    # Настройки для лучшего взаимодействия на мобильных устройствах
    config = {
        "displayModeBar": True,
        "responsive": True,
        "scrollZoom": True,
        "doubleClick": "reset",  # Двойной тап для сброса вида
        "modeBarButtonsToRemove": ["select2d", "lasso2d", "toggleSpikelines"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": "family_tree",
            "scale": 2  # Увеличиваем разрешение изображений для экспорта
        }
    }
    
    # Для мобильных устройств добавляем меньше элементов управления
    if is_mobile:
        config["modeBarButtonsToRemove"].extend(["hoverCompareCartesian", "hoverClosestCartesian"])
    
    return fig

# Функции для определения мобильного устройства
def is_mobile_device():
    """Определяет, запущено ли приложение на мобильном устройстве"""
    try:
        # Пытаемся получить информацию об устройстве из заголовков запроса
        import user_agent
        ua_string = st.session_state.get('user_agent', None)
        if ua_string:
            return user_agent.parse(ua_string).is_mobile
    except:
        pass
        
    # Если не удалось определить - используем JavaScript для проверки размера экрана
    mobile_detector_js = """
    <script>
        // Функция для определения мобильного устройства по размеру экрана и User Agent
        function detectMobile() {
            const mobileWidth = 768;
            const isMobileByWidth = window.innerWidth <= mobileWidth;
            const isMobileByUA = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
            
            // Сохраняем результат в локальное хранилище
            localStorage.setItem('isMobile', (isMobileByWidth || isMobileByUA));
            
            // Передаем информацию компоненту Streamlit через сообщения
            if (window.parent) {
                window.parent.postMessage({
                    type: "streamlit:setComponentValue",
                    value: { isMobile: (isMobileByWidth || isMobileByUA) }
                }, "*");
            }
        }
        
        // Вызываем при загрузке страницы
        detectMobile();
        
        // И при изменении размера окна
        window.addEventListener('resize', detectMobile);
    </script>
    """
    
    st.markdown(mobile_detector_js, unsafe_allow_html=True)
    
    # Для тестирования и отладки используем параметр URL
    if "mobile" in st.query_params:
        mobile_value = st.query_params["mobile"]
        if isinstance(mobile_value, list):
            return mobile_value[0].lower() == "true"
        else:
            return mobile_value.lower() == "true"
        
    # По умолчанию предполагаем, что это не мобильное устройство
    return False

# Определяем тип устройства и сохраняем в session_state
if 'is_mobile' not in st.session_state:
    st.session_state.is_mobile = is_mobile_device()

# Словарь отношений (для подписей)
RELATION_NAMES = {
    "father": "Отец",
    "mother": "Мать",
    "son": "Сын",
    "daughter": "Дочь",
    "husband": "Муж",
    "wife": "Жена",
    "brother": "Брат",
    "sister": "Сестра",
    "grandfather": "Дедушка",
    "grandmother": "Бабушка"
}

# Добавляем небольшую метку версии внизу страницы
st.markdown("""
<div style="position: fixed; bottom: 5px; right: 10px; font-size: 0.7rem; opacity: 0.7;">
    Фамильное древо v2.0 - Mobile Ready
</div>
""", unsafe_allow_html=True)

# Функции для работы с данными
def save_family_data(members, relationships):
    """Сохраняет данные о членах семьи и их отношениях в JSON-файлы"""
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    with open(os.path.join(data_dir, "members.json"), "w", encoding="utf-8") as f:
        json.dump(members, f, ensure_ascii=False, indent=4)
    
    with open(os.path.join(data_dir, "relationships.json"), "w", encoding="utf-8") as f:
        json.dump(relationships, f, ensure_ascii=False, indent=4)

def load_family_data():
    """Загружает данные о членах семьи и их отношениях из JSON-файлов"""
    data_dir = "data"
    members = []
    relationships = []
    
    if os.path.exists(data_dir):
        members_file = os.path.join(data_dir, "members.json")
        relationships_file = os.path.join(data_dir, "relationships.json")
        
        if os.path.exists(members_file):
            with open(members_file, "r", encoding="utf-8") as f:
                members = json.load(f)
        
        if os.path.exists(relationships_file):
            with open(relationships_file, "r", encoding="utf-8") as f:
                relationships = json.load(f)
    
    return members, relationships

def build_family_graph(members, relationships):
    """Создает граф семейного древа на основе данных о членах семьи и их отношениях"""
    G = nx.DiGraph()
    
    # Добавляем узлы (членов семьи)
    for member in members:
        G.add_node(member["id"], **member)
    
    # Добавляем связи (родительские отношения)
    for rel in relationships:
        G.add_edge(rel["parent_id"], rel["child_id"])
    
    return G

def check_relationship_validity(members, parent_id, child_id):
    """Проверяет валидность родительской связи"""
    parent = next((m for m in members if m["id"] == parent_id), None)
    child = next((m for m in members if m["id"] == child_id), None)
    
    if not parent or not child:
        return False, "Один из членов семьи не найден"
    
    # Проверка возраста (родитель должен быть старше ребенка)
    if parent["birth_year"] >= child["birth_year"]:
        return False, f"Родитель ({parent['name']}) должен быть старше ребенка ({child['name']})"
    
    # Проверка на циклические связи
    G = build_family_graph(members, [{"parent_id": parent_id, "child_id": child_id}])
    
    # Проверяем, является ли "ребенок" уже предком "родителя"
    def is_ancestor(graph, ancestor, descendant):
        for successor in graph.successors(ancestor):
            if successor == descendant:
                return True
            if is_ancestor(graph, successor, descendant):
                return True
        return False
    
    if is_ancestor(G, child_id, parent_id):
        return False, "Обнаружена циклическая связь в древе"
    
    return True, ""

def find_member_by_id(members, member_id):
    """Находит члена семьи по ID"""
    for member in members:
        if member["id"] == member_id:
            return member
    return None

def get_relation_to_georgy(members, relationships, person_id):
    """
    Определяет, кем человек с указанным ID является по отношению к Георгию Богданову (ID=3)
    """
    # ID Георгия Богданова
    georgy_id = 3
    
    # Если это сам Георгий
    if person_id == georgy_id:
        return "Это я"
    
    # Строим граф для анализа отношений
    G = nx.DiGraph()
    for member in members:
        G.add_node(member["id"], **member)
    for rel in relationships:
        G.add_edge(rel["parent_id"], rel["child_id"])
    
    # Находим прямых родителей Георгия
    parents_of_georgy = list(G.predecessors(georgy_id))
    
    # Находим прямых детей Георгия (если есть)
    children_of_georgy = list(G.successors(georgy_id))
    
    # Находим братьев/сестер Георгия (имеют тех же родителей)
    siblings = []
    for parent_id in parents_of_georgy:
        for child_id in G.successors(parent_id):
            if child_id != georgy_id and child_id not in siblings:
                siblings.append(child_id)
    
    # Проверяем родительские связи
    if person_id in parents_of_georgy:
        person = next(m for m in members if m["id"] == person_id)
        return "Отец" if person["gender"] == "Мужской" else "Мать"
    
    # Проверяем, является ли человек ребенком Георгия
    if person_id in children_of_georgy:
        person = next(m for m in members if m["id"] == person_id)
        return "Сын" if person["gender"] == "Мужской" else "Дочь"
    
    # Проверяем, является ли человек братом/сестрой Георгия
    if person_id in siblings:
        person = next(m for m in members if m["id"] == person_id)
        return "Брат" if person["gender"] == "Мужской" else "Сестра"
    
    # Проверяем бабушек/дедушек (родители родителей)
    for parent_id in parents_of_georgy:
        grandparents = list(G.predecessors(parent_id))
        if person_id in grandparents:
            person = next(m for m in members if m["id"] == person_id)
            return "Дедушка" if person["gender"] == "Мужской" else "Бабушка"
    
    # Проверяем дядей/теть (братья/сестры родителей)
    uncles_aunts = []
    for parent_id in parents_of_georgy:
        parent_parents = list(G.predecessors(parent_id))
        for grandparent in parent_parents:
            for uncle_aunt in G.successors(grandparent):
                if uncle_aunt != parent_id and uncle_aunt not in uncles_aunts:
                    uncles_aunts.append(uncle_aunt)
    
    if person_id in uncles_aunts:
        person = next(m for m in members if m["id"] == person_id)
        return "Дядя" if person["gender"] == "Мужской" else "Тетя"
    
    # Проверяем двоюродных братьев/сестер (дети дядей/теть)
    cousins = []
    for uncle_aunt in uncles_aunts:
        for cousin in G.successors(uncle_aunt):
            if cousin not in cousins:
                cousins.append(cousin)
    
    if person_id in cousins:
        person = next(m for m in members if m["id"] == person_id)
        return "Двоюродный брат" if person["gender"] == "Мужской" else "Двоюродная сестра"
    
    # По умолчанию, если отношение не определено
    return "Родственник"

# CSS для оформления интерфейса
st.markdown("""
<style>
    /* Стиль для верхних вкладок */
    .top-buttons {
        display: flex;
        margin-bottom: 10px;
    }
    
    .top-buttons button {
        flex: 1;
        height: 50px;
        font-size: 16px !important;
    }
    
    /* Стили для улучшения внешнего вида древа */
    .stPlotlyChart {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Улучшение стилей аккордеона */
    .streamlit-expanderHeader {
        background-color: #f1f3f4;
        border-radius: 5px;
    }
    
    /* Улучшение стилей карточек */
    .member-card {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid;
        transition: all 0.2s ease;
    }
    
    .member-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    /* Убираем лишние отступы */
    .main .block-container {
        padding-top: 1rem !important;
    }
    
    /* Адаптивные стили для мобильных устройств */
    @media (max-width: 768px) {
        /* Уменьшаем отступы на мобильных устройствах */
        .main .block-container {
            padding: 0.5rem !important;
            margin: 0 !important;
        }
        
        /* Улучшаем кнопки навигации для тач-интерфейса */
        .top-buttons button {
            height: 60px;
            font-size: 18px !important;
            padding: 10px 5px !important;
        }
        
        /* Изменяем стиль карточек для лучшей читаемости на мобильных */
        .member-card {
            padding: 0.8rem;
            margin-bottom: 0.8rem;
        }
        
        /* Увеличиваем размер текста для лучшей читаемости */
        .stMarkdown p, .stSelectbox, .stNumberInput, .stTextInput {
            font-size: 16px !important;
        }
        
        /* Кнопки действий больше для тач-интерфейса */
        button {
            min-height: 44px !important;
        }
        
        /* Корректируем размер графика */
        .stPlotlyChart {
            height: calc(100vh - 150px) !important;
            padding: 5px;
            border-radius: 8px;
        }
        
        /* Улучшаем отображение вкладок */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: normal !important;
            padding: 5px !important;
        }
    }
    
    /* Современный эстетический стиль для приложения */
    body {
        background-color: #f9f9f9;
        color: #333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 600;
    }
    
    /* Улучшаем внешний вид заголовков */
    h1 {
        font-size: 1.8rem !important;
        margin-bottom: 1rem !important;
    }
    
    h2 {
        font-size: 1.5rem !important;
    }
    
    h3 {
        font-size: 1.2rem !important;
    }
    
    /* Современные кнопки */
    button[kind="primary"] {
        background-color: #4361ee !important;
    }
    
    /* Улучшенные карточки */
    .modern-card {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        padding: 15px;
        margin-bottom: 15px;
        transition: transform 0.3s, box-shadow 0.3s;
        border-top: 4px solid transparent;
    }
    
    .modern-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    .modern-card-male {
        border-top-color: #4361ee;
    }
    
    .modern-card-female {
        border-top-color: #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)

# Инициализация состояния приложения
if 'force_reset' not in st.session_state:
    # Сброс всех данных при запуске приложения
    st.session_state.clear()
    st.session_state.force_reset = True

if 'members' not in st.session_state:
    # Всегда загружаем структуру из предопределенного JSON
    members = [
        # Основные родители
        {"id": 1, "name": "Мария Ивановна Богданова", "birth_year": 1980, "gender": "Женский"},
        {"id": 2, "name": "Юрий Вячеславович Богданов", "birth_year": 1978, "gender": "Мужской"},
        
        # Дети основных родителей
        {"id": 3, "name": "Георгий Юрьевич Богданов", "birth_year": 2005, "gender": "Мужской"},
        {"id": 4, "name": "Ярослава Юрьевна Богданова", "birth_year": 2007, "gender": "Женский"},
        
        # Родители Марии
        {"id": 5, "name": "Татьяна Сергеевна Шаньшерова", "birth_year": 1960, "gender": "Женский"},
        {"id": 6, "name": "Иван Петрович Шаньшеров", "birth_year": 1958, "gender": "Мужской"},
        
        # Братья и сестры Татьяны
        {"id": 7, "name": "Наталья Хомякова", "birth_year": 1962, "gender": "Женский"},
        {"id": 8, "name": "Алексей Шишкин", "birth_year": 1964, "gender": "Мужской"},
        
        # Братья и сестры Ивана
        {"id": 9, "name": "Леонид Шаньшеров", "birth_year": 1960, "gender": "Мужской"},
        {"id": 10, "name": "Ольга Шаньшерова", "birth_year": 1962, "gender": "Женский"},
        {"id": 11, "name": "Валентина Щербакова", "birth_year": 1964, "gender": "Женский"},
        
        # Сестра Марии и ее семья
        {"id": 12, "name": "Наталья Ивановна Овчинникова", "birth_year": 1982, "gender": "Женский"},
        {"id": 13, "name": "Андрей Овчинников", "birth_year": 1980, "gender": "Мужской"}, # Предполагаемый муж
        {"id": 14, "name": "Ян Андреевич Овчинников", "birth_year": 2005, "gender": "Мужской"},
        {"id": 15, "name": "Богдан Андреевич Овчинников", "birth_year": 2007, "gender": "Мужской"},
        
        # Родители Юрия
        {"id": 16, "name": "Светлана Михайловна Жижина", "birth_year": 1956, "gender": "Женский"},
        {"id": 17, "name": "Вячеслав Терентьевич Жижин", "birth_year": 1954, "gender": "Мужской"},
        
        # Братья и сестры Юрия
        {"id": 18, "name": "Вячеслав Вячеславович Жижин", "birth_year": 1976, "gender": "Мужской"},
        {"id": 19, "name": "Евгения Вячеславовна Жижина", "birth_year": 1980, "gender": "Женский"},
        {"id": 20, "name": "Сергей", "birth_year": 1978, "gender": "Мужской"}, # Предполагаемый муж Евгении
        
        # Дети Евгении
        {"id": 21, "name": "Полина Сергеева", "birth_year": 2006, "gender": "Женский"},
        {"id": 22, "name": "София Сергеева", "birth_year": 2008, "gender": "Женский"},
    ]

    relationships = [
        # Связи детей с родителями
        {"parent_id": 1, "child_id": 3},  # Мария -> Георгий
        {"parent_id": 1, "child_id": 4},  # Мария -> Ярослава
        {"parent_id": 2, "child_id": 3},  # Юрий -> Георгий
        {"parent_id": 2, "child_id": 4},  # Юрий -> Ярослава
        
        # Связи Марии с родителями
        {"parent_id": 5, "child_id": 1},  # Татьяна -> Мария
        {"parent_id": 6, "child_id": 1},  # Иван -> Мария
        
        # Связь сестры Марии с родителями
        {"parent_id": 5, "child_id": 12},  # Татьяна -> Наталья Овчинникова (предположительно)
        {"parent_id": 6, "child_id": 12},  # Иван -> Наталья Овчинникова
        
        # Связи детей Натальи Овчинниковой
        {"parent_id": 12, "child_id": 14},  # Наталья -> Ян
        {"parent_id": 12, "child_id": 15},  # Наталья -> Богдан
        {"parent_id": 13, "child_id": 14},  # Андрей -> Ян
        {"parent_id": 13, "child_id": 15},  # Андрей -> Богдан
        
        # Связи Юрия с родителями
        {"parent_id": 16, "child_id": 2},  # Светлана -> Юрий
        {"parent_id": 17, "child_id": 2},  # Вячеслав -> Юрий
        
        # Связи братьев/сестер Юрия с родителями
        {"parent_id": 16, "child_id": 18},  # Светлана -> Вячеслав (сын)
        {"parent_id": 17, "child_id": 18},  # Вячеслав -> Вячеслав (сын)
        {"parent_id": 16, "child_id": 19},  # Светлана -> Евгения
        {"parent_id": 17, "child_id": 19},  # Вячеслав -> Евгения
        
        # Связи детей Евгении
        {"parent_id": 19, "child_id": 21},  # Евгения -> Полина
        {"parent_id": 19, "child_id": 22},  # Евгения -> София
        {"parent_id": 20, "child_id": 21},  # Сергей -> Полина
        {"parent_id": 20, "child_id": 22},  # Сергей -> София
    ]
    
    # Сохраняем данные для дальнейшего использования
    save_family_data(members, relationships)
    
    st.session_state.members = members
    st.session_state.relationships = relationships
    st.session_state.next_id = max([m["id"] for m in members]) + 1 if members else 1
    st.session_state.confirm_delete = False
    st.session_state.member_to_delete = None
    st.session_state.show_validation_error = False
    st.session_state.validation_error = ""

# Создаем уникальные ключи для вкладок
if 'tab_key' not in st.session_state:
    st.session_state.tab_key = "tree"  # По умолчанию открываем древо

# Создаем кнопки навигации сверху в более мобильном стиле
st.markdown('<div class="top-buttons">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("⚙️ Настройки", use_container_width=True, key="settings_button", 
                help="Настройки отображения древа"):
        st.session_state.tab_key = "settings"
        st.rerun()
with col2:
    if st.button("🌳 Древо", use_container_width=True, key="tree_button",
                help="Просмотр семейного древа"):
        st.session_state.tab_key = "tree"
        st.rerun()
with col3:
    if st.button("✏️ Редактор", use_container_width=True, key="editor_button",
                help="Добавление и редактирование членов семьи"):
        st.session_state.tab_key = "editor"
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# Выбранная вкладка
current_tab = st.session_state.tab_key

# Добавляем мобильные подсказки в зависимости от выбранной вкладки
if current_tab == "tree":
    st.markdown('<div style="text-align: center; font-size: 0.8rem; margin-bottom: 10px; color: #666;">👉 Используйте два пальца для масштабирования</div>', unsafe_allow_html=True)
elif current_tab == "editor":
    st.markdown('<div style="text-align: center; font-size: 0.8rem; margin-bottom: 10px; color: #666;">✏️ Редактируйте данные о своей семье</div>', unsafe_allow_html=True)
elif current_tab == "settings":
    st.markdown('<div style="text-align: center; font-size: 0.8rem; margin-bottom: 10px; color: #666;">⚙️ Настройка отображения древа</div>', unsafe_allow_html=True)

# Отображаем содержимое в зависимости от выбранной вкладки
if current_tab == "settings":
    # Вкладка 1: Настройки древа
    st.header("Настройки древа")
    
    # Настройки отображения древа в адаптивном дизайне
    st.markdown("""
    <div class="modern-card">
        <h3 style="margin-top:0;">Настройки визуализации</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Получаем текущие значения из state или устанавливаем значения по умолчанию
    current_zoom = st.session_state.get('zoom_level', 100)
    current_spacing = st.session_state.get('node_spacing', 3)
    current_scheme = st.session_state.get('color_scheme', "standard")
    current_show_relations = st.session_state.get('show_relations', True)
    current_show_names = st.session_state.get('show_names', True)
    
    # Адаптивное расположение элементов в зависимости от ширины экрана
    # На мобильных - один столбец, на десктопах - два
    is_mobile = False
    try:
        import user_agent
        ua_string = st.session_state.get('user_agent', None)
        if ua_string and user_agent.parse(ua_string).is_mobile:
            is_mobile = True
    except ImportError:
        pass
    
    if is_mobile:
        # Для мобильных - вертикальное расположение с компактными элементами
        st.markdown("""
        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin-bottom:20px;">
            <p style="margin-bottom:10px; font-weight:bold;">Основные настройки</p>
        </div>
        """, unsafe_allow_html=True)
        
        zoom_level = st.slider("Масштаб", 50, 150, current_zoom, 5, format="%d%%")
        node_spacing = st.slider("Расстояние между узлами", 1, 5, current_spacing, 1)
        
        col1, col2 = st.columns(2)
        with col1:
            show_names = st.checkbox("Имена", current_show_names)
        with col2:
            show_relations = st.checkbox("Связи", current_show_relations)
            
        st.markdown("""
        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin:20px 0;">
            <p style="margin-bottom:10px; font-weight:bold;">Цветовая схема</p>
        </div>
        """, unsafe_allow_html=True)
        
        color_scheme = st.radio("", ["standard", "contrast", "monochrome"], 
                                index=["standard", "contrast", "monochrome"].index(current_scheme),
                                horizontal=True,
                                format_func=lambda x: {"standard": "Стандартная", 
                                                      "contrast": "Контрастная",
                                                      "monochrome": "Монохромная"}[x])
    else:
        # Для десктопов - двухколоночное расположение
        col1, col2 = st.columns(2)
        
        with col1:
            zoom_level = st.slider("Масштаб по умолчанию", 50, 150, current_zoom, 5, format="%d%%")
            node_spacing = st.slider("Расстояние между узлами", 1, 5, current_spacing, 1)
        
        with col2:
            show_names = st.checkbox("Показывать имена", current_show_names)
            show_relations = st.checkbox("Показывать родственные связи", current_show_relations)
        
        color_scheme = st.radio("Цветовая схема", ["standard", "contrast", "monochrome"], 
                                index=["standard", "contrast", "monochrome"].index(current_scheme),
                                horizontal=True,
                                format_func=lambda x: {"standard": "Стандартная", 
                                                      "contrast": "Контрастная",
                                                      "monochrome": "Монохромная"}[x])
    
    # Предпросмотр цветов схемы - адаптивный дизайн
    st.markdown('<h3 style="margin:20px 0 10px 0;">Предпросмотр</h3>', unsafe_allow_html=True)
    
    # Создаем адаптивное отображение предпросмотра схемы
    if is_mobile:
        # Для мобильных - компактное отображение
        col1, col2 = st.columns(2)
        male_colors = []
        female_colors = []
        
        for i in range(2):  # Показываем только два основных уровня
            male_color = get_node_color("Мужской", i, color_scheme)
            female_color = get_node_color("Женский", i, color_scheme)
            male_colors.append(male_color)
            female_colors.append(female_color)
        
        with col1:
            st.markdown(f"""
            <div style="text-align:center; margin-bottom:15px;">
                <div style="background-color: {male_colors[0]}; height: 30px; border-radius: 5px; margin-bottom:5px;"></div>
                <div style="font-size:0.8rem;">Мужчина (центр)</div>
            </div>
            <div style="text-align:center;">
                <div style="background-color: {male_colors[1]}; height: 30px; border-radius: 5px; margin-bottom:5px;"></div>
                <div style="font-size:0.8rem;">Мужчина (1 круг)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="text-align:center; margin-bottom:15px;">
                <div style="background-color: {female_colors[0]}; height: 30px; border-radius: 5px; margin-bottom:5px;"></div>
                <div style="font-size:0.8rem;">Женщина (центр)</div>
            </div>
            <div style="text-align:center;">
                <div style="background-color: {female_colors[1]}; height: 30px; border-radius: 5px; margin-bottom:5px;"></div>
                <div style="font-size:0.8rem;">Женщина (1 круг)</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Для десктопов - полное отображение
        col1, col2, col3, col4 = st.columns(4)
        male_colors = []
        female_colors = []
        
        for i in range(4):
            male_color = get_node_color("Мужской", i, color_scheme)
            female_color = get_node_color("Женский", i, color_scheme)
            male_colors.append(male_color)
            female_colors.append(female_color)
        
        with col1:
            st.markdown(f"<div style='background-color: {male_colors[0]}; height: 30px; border-radius: 5px;'></div>", unsafe_allow_html=True)
            st.caption("Мужчина (центр)")
        
        with col2:
            st.markdown(f"<div style='background-color: {female_colors[0]}; height: 30px; border-radius: 5px;'></div>", unsafe_allow_html=True)
            st.caption("Женщина (центр)")
        
        with col3:
            st.markdown(f"<div style='background-color: {male_colors[1]}; height: 30px; border-radius: 5px;'></div>", unsafe_allow_html=True)
            st.caption("Мужчина (1 круг)")
        
        with col4:
            st.markdown(f"<div style='background-color: {female_colors[1]}; height: 30px; border-radius: 5px;'></div>", unsafe_allow_html=True)
            st.caption("Женщина (1 круг)")
    
    # Кнопка сохранения - адаптивная на полный экран
    st.markdown('<div style="margin-top:25px;"></div>', unsafe_allow_html=True)
    save_button = st.button("Сохранить настройки", use_container_width=True, type="primary")
    if save_button:
        st.session_state.zoom_level = zoom_level
        st.session_state.node_spacing = node_spacing
        st.session_state.color_scheme = color_scheme
        st.session_state.show_relations = show_relations
        st.session_state.show_names = show_names
        
        st.success("Настройки сохранены!")
        
        # Задержка перед обновлением страницы
        st.rerun()

elif current_tab == "tree":
    # Вкладка 2: Древо
    st.header("Фамильное древо")
    
    # Добавляем возможность выбрать центрального человека
    if st.session_state.members:
        # Проверяем размер экрана с помощью JavaScript
        st.markdown("""
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                // Проверяем, насколько узок экран для адаптивной верстки
                var isMobile = window.innerWidth <= 768;
                
                if(isMobile) {
                    // Если мобильный, скрываем неважные элементы
                    document.querySelectorAll('.mobile-optional').forEach(function(el) {
                        el.style.display = 'none';
                    });
                }
            });
        </script>
        """, unsafe_allow_html=True)
        
        # Определяем адаптивный макет в зависимости от устройства
        is_mobile = False
        try:
            import user_agent
            ua_string = st.session_state.get('user_agent', None)
            if ua_string and user_agent.parse(ua_string).is_mobile:
                is_mobile = True
        except ImportError:
            # Если библиотека user_agent не установлена, предполагаем настольный компьютер
            pass
        
        # На мобильных устройствах опции настройки под графиком
        if is_mobile:
            # Сначала отображаем график
            # Создаем визуализацию древа с текущими настройками
            central_person_id = st.session_state.get('central_person_id', 3)  # По умолчанию Георгий Богданов (ID=3)
            show_names = st.session_state.get('show_names', True)
            show_relations = st.session_state.get('show_relations', True)
            color_scheme = st.session_state.get('color_scheme', "standard")
            
            fig = create_concentric_family_tree(
                st.session_state.members, 
                st.session_state.relationships,
                central_person_id=central_person_id,
                show_names=show_names,
                show_relations=show_relations,
                color_scheme=color_scheme
            )
            
            # Отображаем визуализацию
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "responsive": True,
                    "modeBarButtonsToRemove": ["select2d", "lasso2d", "resetScale2d", "toggleSpikelines"]
                })
            
            # Затем под графиком отображаем компактные настройки
            with st.expander("Настройки отображения", expanded=False):
                # Компактный выбор центрального узла
                central_person_idx = st.selectbox(
                    "Центр древа",
                    range(len(st.session_state.members)),
                    format_func=lambda i: f"{st.session_state.members[i]['name']}",
                    index=next((i for i, m in enumerate(st.session_state.members) if m["id"] == central_person_id), 0)
                )
                
                # Опции отображения в одну строку
                col1, col2 = st.columns(2)
                with col1:
                    show_names = st.checkbox("Имена", value=show_names)
                with col2:
                    show_relations = st.checkbox("Связи", value=show_relations)
                
                # Цветовая схема
                color_scheme = st.radio(
                    "Цвета", 
                    ["standard", "contrast", "monochrome"],
                    index=["standard", "contrast", "monochrome"].index(color_scheme),
                    format_func=lambda x: {"standard": "Стандарт", 
                                          "contrast": "Контраст",
                                          "monochrome": "Моно"}[x],
                    horizontal=True
                )
                
                # Кнопка применения настроек
                if st.button("Применить", use_container_width=True):
                    st.session_state.central_person_id = st.session_state.members[central_person_idx]["id"]
                    st.session_state.show_names = show_names
                    st.session_state.show_relations = show_relations
                    st.session_state.color_scheme = color_scheme
                    st.rerun()
        else:
            # На десктопе опции настройки справа от графика
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.subheader("Настройки древа")
                # Выбор центрального узла
                central_person_idx = st.selectbox(
                    "Выберите центр древа",
                    range(len(st.session_state.members)),
                    format_func=lambda i: f"{st.session_state.members[i]['name']}",
                    index=2  # По умолчанию Георгий Богданов (индекс 2)
                )
                
                central_person_id = st.session_state.members[central_person_idx]["id"]
                
                # Опции отображения для быстрого переключения
                show_names = st.checkbox("Показать полные имена", value=st.session_state.get('show_names', True))
                show_relations = st.checkbox("Показать родственные связи", value=st.session_state.get('show_relations', True))
                
                # Выбор цветовой схемы
                color_scheme = st.selectbox(
                    "Цветовая схема", 
                    ["standard", "contrast", "monochrome"],
                    index=["standard", "contrast", "monochrome"].index(st.session_state.get('color_scheme', "standard")),
                    format_func=lambda x: {"standard": "Стандартная", 
                                          "contrast": "Контрастная",
                                          "monochrome": "Монохромная"}[x]
                )
                
                # Сохраняем текущие настройки
                st.session_state.color_scheme = color_scheme
                st.session_state.show_names = show_names
                st.session_state.show_relations = show_relations
                st.session_state.central_person_id = central_person_id
            
            with col1:
                # Создаем визуализацию древа
                fig = create_concentric_family_tree(
                    st.session_state.members, 
                    st.session_state.relationships, 
                    central_person_id=central_person_id,
                    show_names=show_names,
                    show_relations=show_relations,
                    color_scheme=color_scheme
                )
                
                # Отображаем визуализацию
                if fig:
                    st.plotly_chart(fig, use_container_width=True, config={
                        "displayModeBar": True,
                        "scrollZoom": True
                    })
                    
                    # Объяснение условных обозначений
                    with st.expander("Легенда и подсказки"):
                        st.markdown("""
                        ### Как читать фамильное древо:
                        
                        - **Центр древа**: выбранный вами человек
                        - **Цвета узлов**: синий для мужчин, розовый для женщин
                        - **Линии связи**:
                            - **Сплошная линия**: родитель-ребенок
                            - **Пунктирная линия**: супружеские отношения
                        
                        **Концентрические круги**:
                        1. **Первый круг**: сам центральный человек
                        2. **Второй круг**: прямая семья (родители, супруг, дети)
                        3. **Третий круг**: близкие родственники (бабушки/дедушки, братья/сестры)
                        4. **Четвертый круг**: дальние родственники (дяди/тети, двоюродные братья/сестры)
                        
                        #### Взаимодействие:
                        - Наведите мышь на узел для отображения имени и родственной связи
                        - Используйте колесико мыши для масштабирования
                        - Перетаскивайте график для перемещения
                        - Выберите другой центр древа в выпадающем меню справа
                        """)
                else:
                    st.error("Не удалось создать визуализацию древа")
    else:
        st.info("Добавьте членов семьи на вкладке 'Редактор', чтобы построить древо")

elif current_tab == "editor":
    # Вкладка 3: Редактирование древа
    st.header("Редактор древа")
    
    # Создаем подвкладки для добавления и редактирования
    edit_tab1, edit_tab2 = st.tabs(["Добавить", "Редактировать"])
    
    with edit_tab1:
        # Форма добавления в компактном виде для лучшей мобильной поддержки
        with st.form(key='add_member_form'):
            st.markdown('<h3 style="margin-top:0">Новый член семьи</h3>', unsafe_allow_html=True)
            
            new_name = st.text_input("Имя", placeholder="Введите полное имя")
            col1, col2 = st.columns(2)
            with col1:
                # Получаем текущий год
                current_year = datetime.datetime.now().year
                new_birth_year = st.number_input("Год рождения", min_value=1800, max_value=current_year, value=1980, step=1)
            with col2:
                new_gender = st.selectbox("Пол", ["Мужской", "Женский"])
            
            # Выбор родителей из существующих членов - компактное отображение
            st.subheader("Родители")
            existing_members = [f"{m['name']} ({m['birth_year']})" for m in st.session_state.members]
            existing_members.insert(0, "Не выбрано")
            
            # Используем компактное горизонтальное расположение для мобильных
            col1, col2 = st.columns(2)
            with col1:
                parent1_idx = st.selectbox("Родитель 1", range(len(existing_members)), 
                                          format_func=lambda i: existing_members[i], key="parent1")
            with col2:
                parent2_idx = st.selectbox("Родитель 2", range(len(existing_members)), 
                                          format_func=lambda i: existing_members[i], key="parent2")
            
            submit_button = st.form_submit_button(label="Добавить", use_container_width=True)
            
            # Обработка формы при добавлении нового члена семьи
            if submit_button:
                if not new_name or len(new_name.strip()) < 2:
                    st.error("Введите корректное имя (минимум 2 символа)")
                else:
                    # Проверяем уникальность имени
                    if any(m["name"] == new_name and m["birth_year"] == new_birth_year for m in st.session_state.members):
                        st.error(f"Член семьи с именем '{new_name}' и годом рождения {new_birth_year} уже существует")
                    else:
                        # Добавляем нового члена семьи
                        new_member = {
                            "id": st.session_state.next_id,
                            "name": new_name,
                            "birth_year": new_birth_year,
                            "gender": new_gender
                        }
                        
                        valid_relationships = True
                        error_message = ""
                        
                        # Проверяем валидность родительских связей
                        if parent1_idx != 0:  # "Не выбрано" имеет индекс 0
                            parent1 = st.session_state.members[parent1_idx - 1]
                            is_valid, message = check_relationship_validity(
                                st.session_state.members + [new_member],
                                parent1["id"], 
                                new_member["id"]
                            )
                            if not is_valid:
                                valid_relationships = False
                                error_message = message
                        
                        if valid_relationships and parent2_idx != 0:
                            parent2 = st.session_state.members[parent2_idx - 1]
                            is_valid, message = check_relationship_validity(
                                st.session_state.members + [new_member],
                                parent2["id"],
                                new_member["id"]
                            )
                            if not is_valid:
                                valid_relationships = False
                                error_message = message
                        
                        if not valid_relationships:
                            st.error(error_message)
                        else:
                            st.session_state.members.append(new_member)
                            st.session_state.next_id += 1
                            
                            # Добавляем связи с родителями
                            if parent1_idx != 0:
                                parent1 = st.session_state.members[parent1_idx - 1]
                                st.session_state.relationships.append({
                                    "parent_id": parent1["id"],
                                    "child_id": new_member["id"]
                                })
                            
                            if parent2_idx != 0:
                                parent2 = st.session_state.members[parent2_idx - 1]
                                st.session_state.relationships.append({
                                    "parent_id": parent2["id"],
                                    "child_id": new_member["id"]
                                })
                            
                            # Сохраняем данные
                            save_family_data(st.session_state.members, st.session_state.relationships)
                            
                            st.success(f"Добавлен новый член семьи: {new_name}")
                            st.rerun()
    
    with edit_tab2:
        # Редактирование и удаление - адаптивный интерфейс
        if st.session_state.members:
            # Более компактный селектор для мобильных устройств
            selected_member_idx = st.selectbox(
                "Выберите члена семьи для редактирования:", 
                range(len(st.session_state.members)), 
                format_func=lambda i: f"{st.session_state.members[i]['name']} ({st.session_state.members[i]['birth_year']})"
            )
            
            member_info = st.session_state.members[selected_member_idx]
            
            # Создаем современную карточку для информации о члене семьи
            gender_color = "#4361ee" if member_info['gender'] == "Мужской" else "#ff6b6b"
            gender_icon = "♂️" if member_info['gender'] == "Мужской" else "♀️"
            card_class = "modern-card-male" if member_info['gender'] == "Мужской" else "modern-card-female"
            
            st.markdown(f"""
            <div class="modern-card {card_class}">
                <h3 style="margin-top: 0; color: {gender_color};">{gender_icon} {member_info['name']}</h3>
                <p><strong>Год рождения:</strong> {member_info['birth_year']}</p>
                <p><strong>Пол:</strong> {member_info['gender']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Вывод информации о родителях и детях в современных карточках
            col1, col2 = st.columns(2)
            with col1:
                # Родители
                parent_ids = [r["parent_id"] for r in st.session_state.relationships if r["child_id"] == member_info["id"]]
                parents = [m for m in st.session_state.members if m["id"] in parent_ids]
                
                st.markdown('<h4 style="margin-bottom:8px;">Родители:</h4>', unsafe_allow_html=True)
                if parents:
                    for parent in parents:
                        gender_icon = "♂️" if parent['gender'] == "Мужской" else "♀️"
                        parent_color = "#4361ee" if parent['gender'] == "Мужской" else "#ff6b6b"
                        st.markdown(f"""
                        <div style="padding:8px; border-left:3px solid {parent_color}; margin-bottom:5px; 
                                    background-color:{parent_color}15; border-radius:5px;">
                            {gender_icon} {parent['name']} ({parent['birth_year']})
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color:#999; font-style:italic;">Родители не указаны</div>', unsafe_allow_html=True)
            
            with col2:
                # Дети
                child_ids = [r["child_id"] for r in st.session_state.relationships if r["parent_id"] == member_info["id"]]
                children = [m for m in st.session_state.members if m["id"] in child_ids]
                
                st.markdown('<h4 style="margin-bottom:8px;">Дети:</h4>', unsafe_allow_html=True)
                if children:
                    for child in children:
                        gender_icon = "♂️" if child['gender'] == "Мужской" else "♀️"
                        child_color = "#4361ee" if child['gender'] == "Мужской" else "#ff6b6b"
                        st.markdown(f"""
                        <div style="padding:8px; border-left:3px solid {child_color}; margin-bottom:5px; 
                                    background-color:{child_color}15; border-radius:5px;">
                            {gender_icon} {child['name']} ({child['birth_year']})
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color:#999; font-style:italic;">Дети не указаны</div>', unsafe_allow_html=True)
            
            # Кнопки действий
            st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                # Кнопка удаления с более заметным оформлением
                if st.button("🗑️ Удалить", key=f"delete_{member_info['id']}", use_container_width=True):
                    # Проверяем наличие связей
                    child_ids = [r["child_id"] for r in st.session_state.relationships if r["parent_id"] == member_info["id"]]
                    has_children = len(child_ids) > 0
                    
                    parent_ids = [r["parent_id"] for r in st.session_state.relationships if r["child_id"] == member_info["id"]]
                    has_parents = len(parent_ids) > 0
                    
                    if has_children or has_parents:
                        st.warning(f"Вы уверены, что хотите удалить {member_info['name']}? Будут потеряны связи между родителями и детьми этого члена семьи.")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Да, удалить", key=f"confirm_{member_info['id']}", use_container_width=True):
                                # Удаляем члена семьи и все связи с ним
                                st.session_state.members.pop(selected_member_idx)
                                st.session_state.relationships = [
                                    r for r in st.session_state.relationships 
                                    if r["parent_id"] != member_info["id"] and r["child_id"] != member_info["id"]
                                ]
                                save_family_data(st.session_state.members, st.session_state.relationships)
                                st.success(f"Член семьи {member_info['name']} удален")
                                st.rerun()
                        with col2:
                            if st.button("Отмена", use_container_width=True):
                                st.rerun()
                    else:
                        # Удаляем члена семьи (нет связей)
                        st.session_state.members.pop(selected_member_idx)
                        save_family_data(st.session_state.members, st.session_state.relationships)
                        st.success(f"Член семьи {member_info['name']} удален")
                        st.rerun()
            
            with col2:
                # Кнопка возврата к просмотру древа с новым центральным узлом
                if st.button("🌳 Показать в древе", use_container_width=True):
                    st.session_state.tab_key = "tree"
                    st.session_state.central_person_id = member_info["id"]
                    st.session_state.show_names = True
                    st.session_state.show_relations = True
                    st.rerun()
        else:
            st.info("Добавьте членов семьи на вкладке 'Добавить', чтобы редактировать их")

# Обработка выбора вкладки из URL
try:
    if "tab" in st.query_params:
        url_tab = st.query_params["tab"]
        if isinstance(url_tab, list):
            url_tab = url_tab[0]
        if url_tab and url_tab in ["settings", "tree", "editor"] and url_tab != st.session_state.tab_key:
            st.session_state.tab_key = url_tab
            st.rerun()
except:
    pass
