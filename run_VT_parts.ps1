$explainers = 'gs', 'l', 'o', 'svs', 'fa', 'ks'
foreach ($explainer in $explainers)
{
    for (($i = 0); $i -lt 50; $i++)
    {
        Measure-Command {
            # C:/Users/jebcu/anaconda3/envs/cs471_project/python.exe VT_generate_preds.py $i
            C:/Users/jebcu/anaconda3/envs/cs471_project/python.exe VT_generate_explanations.py $i $explainer
        }
    }
}