"""Scenario accounting, NOT economics inferred from tiny byte models.
All USD figures are user-specified price targets or explicit assumptions.
"""
import json
from pathlib import Path
RATES={'uncached_input':.05,'cached_input':.001,'generated_output':.10}
# Fixed illustrative workload across every comparison; output includes reasoning.
WORKLOAD={'uncached_input':10_000_000,'cached_input':5_000_000,'visible_output':2_000_000,'hidden_output':2_000_000}
def retail(w):
 return (w['uncached_input']*RATES['uncached_input']+w['cached_input']*RATES['cached_input']+(w['visible_output']+w['hidden_output'])*RATES['generated_output'])/1e6
def serving_budget(gross,tax,margin,other):
 return gross/(1+tax)*(1-margin)-sum(other.values())
def output_tps_required(hourly,utilization,output_price,margin=0):
 # phase-only allocated accelerator cost. Other expense makes requirement harder.
 return hourly/(3600*utilization)/(output_price*(1-margin)/1e6)
def main():
 other={'payment_processing':15,'staff_support_allocation':40,'training_research_amortization':60,'tools_storage_network':20,'contingency':15}
 result={'scope':'User retail targets and illustrative accounting only. No measured frontier-quality cost, utilization, demand, or profitability. Byte-model CPU latency cannot be converted to provider subword-token costs.','price_target_usd_per_million':RATES,'price_target_source':'User-specified half of Muse Spark 1.3 Contributor rates, not independently verified market pricing','fixed_illustrative_monthly_workload':WORKLOAD,'target_retail_bill_usd':retail(WORKLOAD),'subscription_gross_usd':500,'nonserving_cost_assumptions_usd':other,'subscription_scenarios':[{'tax_fraction':tax,'target_margin_on_net_sales':margin,'maximum_serving_cost_usd':serving_budget(500,tax,margin,other)} for tax in [0,.2] for margin in [0,.3,.7]],'phase_only_required_output_tps':[{'illustrative_accelerator_usd_hour':h,'utilization':u,'target_output_price_usd_million':.1,'minimum_busy_aggregate_output_tps_before_other_costs':output_tps_required(h,u,.1)} for h in [.25,1,4] for u in [.3,.6]],'gates':['Measure quality equivalence and fixed actual user workload including hidden output before claiming savings.','Measure real service p95 latency, batching, concurrency, paid hardware cost, utilization, redundancy, and cache costs.','Add training amortization, failed experiments, staffing, tools, taxes and payment expenses.','A zero-margin budget is break-even, not sustainable positive profit.','No change to allowance or quality is credited. Real user workload is not yet specified; this ledger is only a scenario.']}
 Path(__file__).with_name('economics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
