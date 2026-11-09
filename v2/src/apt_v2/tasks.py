"""Deterministic business datasets with independently perturbed verifier worlds."""
from dataclasses import dataclass,asdict
import random
import sqlite3

SCHEMA = """orders(order_id INTEGER, order_month TEXT, region TEXT, segment TEXT,
 status TEXT, gross_cents INTEGER);
refunds(refund_id INTEGER, order_id INTEGER, refund_month TEXT, refund_cents INTEGER)."""
POLICY = (
 "Amounts are integer cents. Count gross revenue from paid orders in the requested month, region and segment. "
 "Subtract refunds occurring in that month for paid orders of the same region and segment, "
 "even when the original order is from another month. Exclude cancelled orders and their refunds. "
 "Net revenue = eligible gross - eligible refunds. Change = current net - prior net. Policy ID: revenue-v1."
)

@dataclass(frozen=True)
class Task:
    id:str
    split:str
    family_id:str
    seed:int
    current_month:str
    prior_month:str
    region:str
    segment:str
    fault_mode:str='none'
    max_turns:int=12

    def public(self):
        return {k:v for k,v in asdict(self).items() if k not in ('seed','split','family_id','fault_mode')}

    def request(self):
        return (f"Reconcile net revenue for region {self.region}, segment {self.segment}. "
            f"Current month: {self.current_month}; prior month: {self.prior_month}. "
            "Read the business policy and schema, calculate current_net_cents, prior_net_cents, "
            "and change_cents; submit all three integer values with query evidence IDs and policy ID. "
            "Inspect failed checks and repair errors before submitting.")

def make_tasks():
    result=[]
    for split,offset,count in [('train',1000,12),('development',2000,4),('test',3000,4)]:
        for i in range(count):
            month=3+i%7
            result.append(Task(f'{split}-{i:03}',split,f'family-{offset+i}',offset+i,
                f'2026-{month:02}',f'2026-{month-1:02}',
                ['North','West','South'][i%3],f'Segment-{offset+i}',
                'transient_query' if i%2 else 'none'))
    return result

def connect(task,variant=0):
    rng=random.Random(task.seed)
    db=sqlite3.connect(':memory:')
    db.executescript("""CREATE TABLE orders(order_id INTEGER PRIMARY KEY,order_month TEXT,region TEXT,
        segment TEXT,status TEXT,gross_cents INTEGER);
        CREATE TABLE refunds(refund_id INTEGER PRIMARY KEY,order_id INTEGER,refund_month TEXT,refund_cents INTEGER);""")
    orders=[];refunds=[]
    months=[task.current_month,task.prior_month,'2025-12']
    for month in months:
        for region in (task.region,'Other'):
            for segment in (task.segment,'Other'):
                for status in ('paid','cancelled'):
                    for _ in range(3):
                        oid=len(orders)+1
                        amount=rng.randint(1000,12000)
                        orders.append((oid,month,region,segment,status,amount))
                        if oid%2==0:
                            refunds.append((len(refunds)+1,oid,rng.choice(months),rng.randint(100,amount//2)))
    # Perturb all numeric amounts independently, preserving joins and business rules.
    if variant:
        prng=random.Random(task.seed+variant*1000003)
        orders=[(*r[:-1],prng.randint(500,20000)) for r in orders]
        refunds=[(*r[:-1],prng.randint(50,5000)) for r in refunds]
    db.executemany('INSERT INTO orders VALUES(?,?,?,?,?,?)',orders)
    db.executemany('INSERT INTO refunds VALUES(?,?,?,?)',refunds)
    db.commit()
    return db

def metric_sql(task,metric):
    if metric not in ('current_net_cents','prior_net_cents'):
        raise ValueError('a monthly net metric is required')
    month=task.current_month if metric=='current_net_cents' else task.prior_month
    quote=lambda value:"'"+value.replace("'","''")+"'"
    return (
        f"SELECT (SELECT COALESCE(SUM(gross_cents),0) FROM orders WHERE status='paid' "
        f"AND order_month={quote(month)} AND region={quote(task.region)} AND segment={quote(task.segment)}) "
        f"- (SELECT COALESCE(SUM(r.refund_cents),0) FROM refunds r JOIN orders o ON r.order_id=o.order_id "
        f"WHERE o.status='paid' AND r.refund_month={quote(month)} AND o.region={quote(task.region)} "
        f"AND o.segment={quote(task.segment)}) AS {metric}"
    )

def truth(task,variant=0):
    db=connect(task,variant)
    try:
        cur=db.execute(metric_sql(task,'current_net_cents')).fetchone()[0]
        prior=db.execute(metric_sql(task,'prior_net_cents')).fetchone()[0]
        return dict(current_net_cents=cur,prior_net_cents=prior,change_cents=cur-prior)
    finally:db.close()
