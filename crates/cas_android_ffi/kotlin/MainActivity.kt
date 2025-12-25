package es.javiergimenez.explicas

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.Spinner
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

/**
 * Demo Activity for ExpliCAS engine integration.
 * 
 * Uses JSON schema v1 from cas_engine::json.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var editExpression: EditText
    private lateinit var btnEvaluate: Button
    private lateinit var txtResult: TextView
    private lateinit var txtStatus: TextView
    private lateinit var spinnerPreset: Spinner
    private lateinit var switchStrict: SwitchCompat
    private lateinit var switchShowSteps: SwitchCompat
    private lateinit var txtAbiVersion: TextView
    private lateinit var recyclerSteps: RecyclerView
    private lateinit var stepsAdapter: StepsAdapter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Bind views
        editExpression = findViewById(R.id.editExpression)
        btnEvaluate = findViewById(R.id.btnEvaluate)
        txtResult = findViewById(R.id.txtResult)
        txtStatus = findViewById(R.id.txtTimings) // Reusing as status
        spinnerPreset = findViewById(R.id.spinnerPreset)
        switchStrict = findViewById(R.id.switchStrict)
        switchShowSteps = findViewById(R.id.switchShowAllSteps)
        txtAbiVersion = findViewById(R.id.txtAbiVersion)
        recyclerSteps = findViewById(R.id.recyclerSteps)

        // Setup RecyclerView
        stepsAdapter = StepsAdapter()
        recyclerSteps.layoutManager = LinearLayoutManager(this)
        recyclerSteps.adapter = stepsAdapter

        // Setup preset spinner
        val presets = arrayOf("cli", "small", "unlimited")
        spinnerPreset.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, presets)

        // Show ABI version
        txtAbiVersion.text = "Engine ABI: ${CasNative.abiVersion()}"

        // Evaluate button
        btnEvaluate.setOnClickListener {
            val expr = editExpression.text.toString()
            if (expr.isBlank()) {
                txtResult.text = "Enter an expression"
                return@setOnClickListener
            }
            evaluate(expr)
        }

        // Set default expression for testing
        editExpression.setText("x^2 + 2*x + 1")
    }

    private fun evaluate(expr: String) {
        val preset = spinnerPreset.selectedItem.toString()
        val mode = if (switchStrict.isChecked) "strict" else "best-effort"
        val includeSteps = switchShowSteps.isChecked

        txtResult.text = "Evaluating..."
        txtStatus.text = ""
        stepsAdapter.setSteps(emptyList())
        btnEvaluate.isEnabled = false

        lifecycleScope.launch(Dispatchers.Default) {
            val evalResult = try {
                evalAndParse(expr, preset, mode, includeSteps)
            } catch (e: Exception) {
                EvalResult.Error("E_INTERNAL", "Exception: ${e.message}")
            }

            withContext(Dispatchers.Main) {
                when (evalResult) {
                    is EvalResult.Success -> {
                        txtResult.text = evalResult.result
                        txtStatus.text = if (evalResult.isPartial) "⚠️ Partial result" else ""
                        stepsAdapter.setSteps(evalResult.steps)
                    }
                    is EvalResult.Error -> {
                        txtResult.text = "${evalResult.code}: ${evalResult.message}"
                        txtStatus.text = ""
                        stepsAdapter.setSteps(emptyList())
                    }
                }
                btnEvaluate.isEnabled = true
            }
        }
    }

    /**
     * Parse JSON response into structured result.
     * Uses schema v1 from cas_engine::json.
     */
    private fun evalAndParse(expr: String, preset: String, mode: String, steps: Boolean): EvalResult {
        val optsJson = """{"budget":{"preset":"$preset","mode":"$mode"},"steps":$steps}"""
        val json = CasNative.evalJson(expr, optsJson)
        val obj = JSONObject(json)

        val schemaVersion = obj.optInt("schema_version", 0)
        if (schemaVersion != 1) {
            return EvalResult.Error("E_SCHEMA", "Unexpected schema version: $schemaVersion")
        }

        val ok = obj.optBoolean("ok", false)

        return if (ok) {
            val result = obj.optString("result", "")
            val stepsArray = obj.optJSONArray("steps") ?: JSONArray()
            val stepsList = parseSteps(stepsArray)
            
            // Check for partial result (budget exceeded in best-effort)
            val budget = obj.optJSONObject("budget")
            val isPartial = budget?.has("exceeded") == true && budget.optJSONObject("exceeded") != null
            
            EvalResult.Success(result, stepsList, isPartial)
        } else {
            val error = obj.optJSONObject("error")
            val code = error?.optString("code", "E_UNKNOWN") ?: "E_UNKNOWN"
            val message = error?.optString("message", "Unknown error") ?: "Unknown error"
            EvalResult.Error(code, message)
        }
    }

    /**
     * Parse steps array from new schema (phase/rule/before/after).
     */
    private fun parseSteps(array: JSONArray): List<Step> {
        val steps = mutableListOf<Step>()
        for (i in 0 until array.length()) {
            val obj = array.getJSONObject(i)
            steps.add(Step(
                index = i + 1,
                phase = obj.optString("phase", ""),
                rule = obj.optString("rule", ""),
                before = obj.optString("before", ""),
                after = obj.optString("after", "")
            ))
        }
        return steps
    }

    // ========================================================================
    // Data classes (matching schema v1)
    // ========================================================================

    sealed class EvalResult {
        data class Success(
            val result: String,
            val steps: List<Step>,
            val isPartial: Boolean = false
        ) : EvalResult()
        
        data class Error(
            val code: String,
            val message: String
        ) : EvalResult()
    }

    data class Step(
        val index: Int,
        val phase: String,
        val rule: String,
        val before: String,
        val after: String
    )

    // ========================================================================
    // RecyclerView Adapter
    // ========================================================================

    inner class StepsAdapter : RecyclerView.Adapter<StepsAdapter.StepViewHolder>() {
        private var steps: List<Step> = emptyList()

        fun setSteps(newSteps: List<Step>) {
            steps = newSteps
            notifyDataSetChanged()
        }

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): StepViewHolder {
            val view = LayoutInflater.from(parent.context)
                .inflate(R.layout.item_step, parent, false)
            return StepViewHolder(view)
        }

        override fun onBindViewHolder(holder: StepViewHolder, position: Int) {
            holder.bind(steps[position])
        }

        override fun getItemCount(): Int = steps.size

        inner class StepViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
            private val txtStepIndex: TextView = itemView.findViewById(R.id.txtStepIndex)
            private val txtRule: TextView = itemView.findViewById(R.id.txtRule)
            private val txtDescription: TextView = itemView.findViewById(R.id.txtDescription)
            private val txtBefore: TextView = itemView.findViewById(R.id.txtBefore)
            private val txtAfter: TextView = itemView.findViewById(R.id.txtAfter)
            private val txtDomainAssumption: TextView = itemView.findViewById(R.id.txtDomainAssumption)

            fun bind(step: Step) {
                txtStepIndex.text = step.index.toString()
                txtRule.text = step.rule
                txtDescription.text = step.phase
                
                // Before/After
                if (step.before.isNotEmpty() && step.after.isNotEmpty()) {
                    txtBefore.text = step.before
                    txtAfter.text = step.after
                    txtBefore.visibility = View.VISIBLE
                    txtAfter.visibility = View.VISIBLE
                } else {
                    txtBefore.visibility = View.GONE
                    txtAfter.visibility = View.GONE
                }

                // Hide domain assumption (not in new schema)
                txtDomainAssumption.visibility = View.GONE
            }
        }
    }
}
