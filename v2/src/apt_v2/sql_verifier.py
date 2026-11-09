"""Versioned semantic SQL worlds shared by training and new evaluation runs."""
from .tasks import connect, metric_sql

VERSION = 'sql_semantics_v2'
WORLDS = ('amount_1', 'amount_2', 'cross_month', 'no_current_refunds',
          'no_prior_refunds', 'no_target_refunds')
METRICS = ('current_net_cents', 'prior_net_cents', 'change_cents')


def world(task, name):
    if name not in WORLDS and name != 'empty_gross':
        raise ValueError('unknown SQL verification world')
    db = connect(task, 1 if name == 'amount_1' else 2 if name == 'amount_2' else 3)
    eligible = "SELECT order_id FROM orders WHERE status='paid' AND region=? AND segment=?"
    params = (task.region, task.segment)
    try:
        if name == 'cross_month':
            for original, occurred in ((task.current_month, task.prior_month), (task.prior_month, task.current_month)):
                row = db.execute("SELECT r.refund_id FROM refunds r JOIN orders o ON r.order_id=o.order_id "
                                 "WHERE o.order_month=? AND o.status='paid' AND o.region=? AND o.segment=? "
                                 "ORDER BY r.refund_id LIMIT 1", (original, *params)).fetchone()
                if row is None:
                    raise ValueError('cross-month world requires eligible refunds for each original month')
                db.execute('UPDATE refunds SET refund_month=? WHERE refund_id=?', (occurred, row[0]))
        elif name in ('no_current_refunds', 'no_prior_refunds', 'no_target_refunds'):
            removed = ((task.current_month, task.prior_month) if name == 'no_target_refunds' else
                       (task.current_month if name == 'no_current_refunds' else task.prior_month,))
            # An outside month removes both target months without changing joins/statuses.
            outside = '1900-01'
            while outside in (task.current_month, task.prior_month):
                outside = str(int(outside[:4])-1) + '-01'
            placeholders = ','.join('?' for _ in removed)
            db.execute("UPDATE refunds SET refund_month=? WHERE refund_month IN (" + placeholders +
                       ") AND order_id IN (" + eligible + ")", (outside, *removed, *params))
        elif name == 'empty_gross':
            # CPU regression fixture; this is separate from the frozen training world list.
            db.execute("UPDATE orders SET order_month='1900-01' WHERE order_month IN (?,?) "
                       "AND order_id IN (" + eligible + ")", (task.current_month, task.prior_month, *params))
        db.commit()
        return db
    except Exception:
        db.close()
        raise


def truth_in(db, task):
    current = db.execute(metric_sql(task, METRICS[0])).fetchone()[0]
    prior = db.execute(metric_sql(task, METRICS[1])).fetchone()[0]
    return {METRICS[0]: current, METRICS[1]: prior, METRICS[2]: current-prior}


def certify(task, sql, rows, reads, gold):
    from .environment import execute_readonly, ToolError
    if len(rows) != 1 or not {'orders', 'refunds'}.issubset(reads):
        return {}
    valid = {key: rows[0][key] for key in METRICS
             if type(rows[0].get(key)) is int and rows[0][key] == gold[key]}
    if not valid:
        return {}
    for name in WORLDS:
        db = world(task, name)
        try:
            expected = truth_in(db, task)
            try:
                actual, _ = execute_readonly(db, sql)
            except ToolError:
                return {}
            valid = {key: value for key, value in valid.items()
                     if len(actual) == 1 and type(actual[0].get(key)) is int
                     and actual[0][key] == expected[key]}
            if not valid:
                break
        finally:
            db.close()
    return valid
