"""Real SQLite dependency: verified reconciliation -> closing row -> next-period balance."""
import json
import random
from dataclasses import asdict
from .tasks import Task,connect,truth,metric_sql,POLICY,SCHEMA
from .environment import Environment,ToolError,execute_readonly
from .runtime import SYSTEM
from .grpo import digest,parse_or_invalid,validate_native

TABLES=('orders','refunds','period_closings','ledger_movements')
EXTRA_SCHEMA='period_closings(closing_id TEXT, period TEXT, region TEXT, segment TEXT, net_cents INTEGER, change_cents INTEGER, evidence_digest TEXT); ledger_movements(movement_id INTEGER, period TEXT, region TEXT, segment TEXT, amount_cents INTEGER).'
CLOSE_SYSTEM=SYSTEM+' Additional tool close_period accepts values, evidence and policy_id as check_result and returns closing_id. Final submit additionally requires next_balance_cents in values/evidence and closing_id.'

def next_month(month):
    year,mon=map(int,month.split('-')); return f'{year+(mon==12):04}-{mon%12+1:02}'

class ClosingTask(Task):
    def request(self):
        return (super().request()+' Then call close_period with those verified values/evidence. '
                f'The next period is {next_month(self.current_month)}. Query its next_balance_cents from '
                'the persisted period_closings row plus the sum of ledger_movements for the next period, '
                'same region/segment. Finally submit all four values/evidence IDs and closing_id.')

def closing_tasks():
    return [ClosingTask(f'closing-development-{i:03}','development',f'closing-family-{8000+i}',8000+i,
                        f'2026-{4+i:02}',f'2026-{3+i:02}',('North','West','South')[i%3],
                        f'Closing-Segment-{8000+i}','transient_query' if i%2 else 'none',16) for i in range(4)]

def closing_db(task,variant=0):
    db=connect(task,variant)
    db.executescript('CREATE TABLE period_closings(closing_id TEXT PRIMARY KEY, period TEXT UNIQUE, region TEXT, segment TEXT, net_cents INTEGER, change_cents INTEGER, evidence_digest TEXT); CREATE TABLE ledger_movements(movement_id INTEGER PRIMARY KEY, period TEXT, region TEXT, segment TEXT, amount_cents INTEGER);')
    rng=random.Random(task.seed+7000001+variant*1000003); rows=[]
    for period in (next_month(task.current_month),'2025-11'):
        for region in (task.region,'Other'):
            for segment in (task.segment,'Other'):
                for _ in range(3): rows.append((len(rows)+1,period,region,segment,rng.randint(-4000,7000)))
    db.executemany('INSERT INTO ledger_movements VALUES(?,?,?,?,?)',rows); db.commit(); return db

def quote(s): return "'"+s.replace("'","''")+"'"

def next_sql(task):
    return (f'SELECT c.net_cents + COALESCE(SUM(m.amount_cents),0) AS next_balance_cents '
            f'FROM period_closings c JOIN ledger_movements m ON m.region=c.region AND m.segment=c.segment '
            f'WHERE c.period={quote(task.current_month)} AND c.region={quote(task.region)} '
            f'AND c.segment={quote(task.segment)} AND m.period={quote(next_month(task.current_month))} GROUP BY c.closing_id')

def expected_next(db,task):
    row=db.execute('SELECT net_cents FROM period_closings WHERE period=? AND region=? AND segment=?',
                   (task.current_month,task.region,task.segment)).fetchone()
    if row is None: return None
    movement=db.execute('SELECT COALESCE(SUM(amount_cents),0) FROM ledger_movements WHERE period=? AND region=? AND segment=?',
                         (next_month(task.current_month),task.region,task.segment)).fetchone()[0]
    return row[0]+movement

class ClosingEnvironment(Environment):
    def __init__(self,task):
        super().__init__(task,'outcome'); self.db.close(); self.db=closing_db(task); self.closing_expected=None
    def valid_closing(self):
        if self.closing_expected is None: return False
        row=self.db.execute('SELECT * FROM period_closings WHERE closing_id=?',(self.closing_expected[0],)).fetchone()
        return row==self.closing_expected
    def _check(self,args):
        result=super()._check(args); values=args.get('values',{}); evidence=args.get('evidence',{})
        value=values.get('next_balance_cents') if isinstance(values,dict) else None
        eid=evidence.get('next_balance_cents') if isinstance(evidence,dict) else None
        closing=self.valid_closing() and args.get('closing_id')==self.closing_expected[0]
        result['checks'].update(closing=closing,next_balance_cents=(closing and type(value) is int
            and value==expected_next(self.db,self.task) and self.evidence.get(eid,{}).get('next_balance_cents')==value))
        result['passed']=all(result['checks'].values()); return result
    def _verify_next(self,sql,rows,reads):
        if not self.valid_closing() or not {'period_closings','ledger_movements'}.issubset(reads) or len(rows)!=1: return {}
        value=rows[0].get('next_balance_cents')
        if type(value) is not int or value!=expected_next(self.db,self.task): return {}
        for variant in (1,2):
            probe=closing_db(self.task,variant)
            try:
                gold=truth(self.task,variant)
                probe.execute('INSERT INTO period_closings VALUES(?,?,?,?,?,?,?)',('probe',self.task.current_month,self.task.region,self.task.segment,gold['current_net_cents'],gold['change_cents'],'verified-probe'))
                test,_=execute_readonly(probe,sql,TABLES)
                if len(test)!=1 or test[0].get('next_balance_cents')!=expected_next(probe,self.task):return {}
            finally: probe.close()
        return {'next_balance_cents':value}
    def _call(self,tool,args):
        if not isinstance(args,dict): raise ToolError('arguments must be an object')
        if tool=='read_policy':
            self.policy_read=True
            return {'policy_id':'revenue-v1','policy':POLICY+' Close the current period using verified monthly values and evidence. Closing writes a persistent row. Next balance = current closing net + signed next-period ledger movements, same region/segment. Final submission must reference closing_id and verified next_balance_cents evidence.','tools':['read_policy','inspect_schema','query_sql','calculate_change','close_period','check_result','submit']}
        if tool=='inspect_schema': self.schema_read=True; return {'schema':SCHEMA+EXTRA_SCHEMA}
        if tool=='close_period':
            if not super()._check(args)['passed']: raise ToolError('verified current/prior/change values and policy/schema are required before closing')
            if self.closing_expected is not None:
                if not self.valid_closing(): raise ToolError('closing row integrity differs')
                return {'closing_id':self.closing_expected[0],'persisted':True,'already_closed':True}
            eid='closing-'+self.task.id
            self.closing_expected=(eid,self.task.current_month,self.task.region,self.task.segment,
                self.gold['current_net_cents'],self.gold['change_cents'],digest({'values':args['values'],'evidence':args['evidence']}))
            self.db.execute('INSERT INTO period_closings VALUES(?,?,?,?,?,?,?)',self.closing_expected); self.db.commit()
            return {'closing_id':eid,'persisted':True,'closed_net_cents':self.gold['current_net_cents']}
        if tool=='query_sql':
            self.query_attempts+=1
            if self.task.fault_mode=='transient_query' and self.query_attempts==1:raise ToolError('temporary query service error; retry the query')
            rows,reads=execute_readonly(self.db,args.get('sql'),TABLES)
            valid=self._certify_sql(args['sql'],rows,reads); valid.update(self._verify_next(args['sql'],rows,reads))
            eid='q'+str(self.query_attempts);self.evidence[eid]=valid;self.verified.update(valid)
            return {'rows':rows,'evidence_id':eid,'verified_metrics':list(valid),'verification':'source and perturbed-data checks'}
        return super()._call(tool,args)

def run_closing(task,policy,sampling,version,seed,tokenizer=None):
    from dataclasses import asdict as convert
    env=ClosingEnvironment(task); messages=[{'role':'system','content':CLOSE_SYSTEM},{'role':'user','content':task.request()}]; rows=[]
    try:
        while not env.done:
            generation=policy(messages); g=convert(generation)
            if tokenizer is not None:validate_native(tokenizer,messages,g,sampling)
            row=env.step(parse_or_invalid(generation.text));row['generation']=g;row['sampling_seed']=seed+len(rows);rows.append(row)
            messages.extend([{'role':'assistant','content':g['text']},{'role':'user','content':json.dumps({'observation':row['observation']})}])
        return {'task':asdict(task),'sampling':sampling,'policy_version':version,'seed':seed,'turns':rows,
                'success':env.success,'closing_valid':env.valid_closing(),'closing_rows':[list(r) for r in env.db.execute('SELECT * FROM period_closings')],
                'next_balance':expected_next(env.db,task),'artifact':rows[-1]['observation'].get('artifact') if env.success else None}
    except Exception as exc:
        exc.episode_evidence={'task':asdict(task),'seed':seed,'policy_version':version,'status':'interrupted','training_admitted':False,'turns':rows,'error':str(exc)}
        raise
    finally:env.close()

def replay_closing(episode,tokenizer=None):
    task=next(t for t in closing_tasks() if t.id==episode['task']['id'])
    if asdict(task)!=episode['task']:raise ValueError('closing task identity differs')
    env=ClosingEnvironment(task);messages=[{'role':'system','content':CLOSE_SYSTEM},{'role':'user','content':task.request()}]
    try:
        for i,row in enumerate(episode['turns']):
            g=row['generation']
            if tokenizer is not None:validate_native(tokenizer,messages,g,episode['sampling'])
            if row['sampling_seed']!=episode['seed']+i:raise ValueError('closing seed schedule differs')
            actual=env.step(parse_or_invalid(g['text']))
            if actual!={k:v for k,v in row.items() if k not in ('generation','sampling_seed')}:raise ValueError('closing environment replay differs')
            messages.extend([{'role':'assistant','content':g['text']},{'role':'user','content':json.dumps({'observation':actual['observation']})}])
        if (not env.done or episode['success']!=env.success or episode['closing_valid']!=env.valid_closing()
                or episode['closing_rows']!=[list(r) for r in env.db.execute('SELECT * FROM period_closings')]
                or episode['next_balance']!=expected_next(env.db,task)
                or episode['artifact']!=(env.records[-1]['observation'].get('artifact') if env.success else None)):
            raise ValueError('closing final artifact differs')
        return len(episode['turns'])
    finally:env.close()


def expert_dependency_proof(task):
    env=ClosingEnvironment(task)
    def call(tool,args=None):
        row=env.step({'tool':tool,'arguments':args or {}})
        if row['tool_error']: raise ValueError(row['tool_error'])
        return row['observation']
    try:
        call('read_policy');call('inspect_schema');evidence={};values={}
        for metric in ('current_net_cents','prior_net_cents'):
            row=env.step({'tool':'query_sql','arguments':{'sql':metric_sql(task,metric)}})
            if row['tool_error']:
                if 'temporary query' not in row['tool_error']:raise ValueError(row['tool_error'])
                observation=call('query_sql',{'sql':metric_sql(task,metric)})
            else:observation=row['observation']
            evidence[metric]=observation['evidence_id'];values[metric]=observation['rows'][0][metric]
        change=call('calculate_change',{'current_evidence':evidence['current_net_cents'],'prior_evidence':evidence['prior_net_cents']})
        values['change_cents']=change['change_cents'];evidence['change_cents']=change['evidence_id']
        args={'values':dict(values),'evidence':dict(evidence),'policy_id':'revenue-v1'}
        closed=call('close_period',args);observation=call('query_sql',{'sql':next_sql(task)})
        values['next_balance_cents']=observation['rows'][0]['next_balance_cents'];evidence['next_balance_cents']=observation['evidence_id']
        final={'values':values,'evidence':evidence,'policy_id':'revenue-v1','closing_id':closed['closing_id']}
        call('check_result',final);call('submit',final)
        original=env.closing_expected;before=expected_next(env.db,task)
        env.db.execute('DELETE FROM period_closings');missing,_=execute_readonly(env.db,next_sql(task),TABLES)
        env.db.execute('INSERT INTO period_closings VALUES(?,?,?,?,?,?,?)',original)
        env.db.execute('UPDATE period_closings SET net_cents=net_cents+123')
        changed,_=execute_readonly(env.db,next_sql(task),TABLES)
        checks={'expert_success':env.success,'closing_persisted':closed['persisted'],
                'removal_blocks_next_query':missing==[],
                'mutation_changes_balance':changed[0]['next_balance_cents']==before+123,
                'mutation_fails_integrity':not env.valid_closing()}
        return {'task_id':task.id,'passed':all(checks.values()),'checks':checks,'expert_records':env.records,'original_closing_row':list(original),'next_balance_cents':before}
    finally:env.close()
