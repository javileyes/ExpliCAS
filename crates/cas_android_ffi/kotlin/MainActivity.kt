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
 * Demo Activity for ExpliCAS engine integration with steps display.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var editExpression: EditText
    private lateinit var btnEvaluate: Button
    private lateinit var txtResult: TextView
    private lateinit var txtTimings: TextView
    private lateinit var spinnerPreset: Spinner
    private lateinit var switchStrict: SwitchCompat
    private lateinit var switchShowAllSteps: SwitchCompat
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
        txtTimings = findViewById(R.id.txtTimings)
        spinnerPreset = findViewById(R.id.spinnerPreset)
        switchStrict = findViewById(R.id.switchStrict)
        switchShowAllSteps = findViewById(R.id.switchShowAllSteps)
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

        // Toggle steps filter
        switchShowAllSteps.setOnCheckedChangeListener { _, _ ->
            stepsAdapter.notifyDataSetChanged()
        }

        // Set default expression for testing
        editExpression.setText("x^2 + 2*x + 1")
    }

    private fun evaluate(expr: String) {
        val preset = spinnerPreset.selectedItem.toString()
        val mode = if (switchStrict.isChecked) "strict" else "best-effort"

        txtResult.text = "Evaluating..."
        txtTimings.text = ""
        stepsAdapter.setSteps(emptyList())
        btnEvaluate.isEnabled = false

        lifecycleScope.launch(Dispatchers.Default) {
            val evalResult = try {
                evalAndParse(expr, preset, mode)
            } catch (e: Exception) {
                EvalResult.Error("Exception: ${e.message}")
            }

            withContext(Dispatchers.Main) {
                when (evalResult) {
                    is EvalResult.Success -> {
                        txtResult.text = evalResult.result
                        txtTimings.text = "(${evalResult.totalUs}µs)"
                        stepsAdapter.setSteps(evalResult.steps, switchShowAllSteps.isChecked)
                    }
                    is EvalResult.Error -> {
                        txtResult.text = evalResult.message
                        txtTimings.text = ""
                        stepsAdapter.setSteps(emptyList())
                    }
                }
                btnEvaluate.isEnabled = true
            }
        }
    }

    /**
     * Parse JSON response into structured result
     */
    private fun evalAndParse(expr: String, preset: String, mode: String): EvalResult {
        val optsJson = """{"budget":{"preset":"$preset","mode":"$mode"}}"""
        val json = CasNative.evalJson(expr, optsJson)
        val obj = JSONObject(json)

        if (obj.optInt("schema_version", 0) != 1) {
            return EvalResult.Error("Unexpected schema version")
        }

        val ok = obj.optBoolean("ok", true)

        return if (ok) {
            val result = obj.optString("result", "NO_RESULT")
            val timings = obj.optJSONObject("timings_us")
            val totalUs = timings?.optLong("total_us", 0) ?: 0
            val stepsArray = obj.optJSONArray("steps") ?: JSONArray()
            val steps = parseSteps(stepsArray)
            EvalResult.Success(result, totalUs, steps)
        } else {
            val error = obj.optJSONObject("error")
            val kind = error?.optString("kind", "Unknown") ?: "Unknown"
            val message = error?.optString("message", "No message") ?: "No message"
            EvalResult.Error("ERROR ($kind): $message")
        }
    }

    private fun parseSteps(array: JSONArray): List<Step> {
        val steps = mutableListOf<Step>()
        for (i in 0 until array.length()) {
            val obj = array.getJSONObject(i)
            steps.add(Step(
                index = obj.optInt("index", i + 1),
                rule = obj.optString("rule", ""),
                description = obj.optString("description", ""),
                before = obj.optString("before", null),
                after = obj.optString("after", null),
                importance = obj.optString("importance", "medium"),
                domainAssumption = obj.optString("domain_assumption", null)
            ))
        }
        return steps
    }

    // ========================================================================
    // Data classes
    // ========================================================================

    sealed class EvalResult {
        data class Success(val result: String, val totalUs: Long, val steps: List<Step>) : EvalResult()
        data class Error(val message: String) : EvalResult()
    }

    data class Step(
        val index: Int,
        val rule: String,
        val description: String,
        val before: String?,
        val after: String?,
        val importance: String,
        val domainAssumption: String?
    ) {
        fun shouldShow(showAll: Boolean): Boolean {
            return showAll || importance in listOf("medium", "high")
        }
    }

    // ========================================================================
    // RecyclerView Adapter
    // ========================================================================

    inner class StepsAdapter : RecyclerView.Adapter<StepsAdapter.StepViewHolder>() {
        private var allSteps: List<Step> = emptyList()
        private var showAll: Boolean = false

        private val filteredSteps: List<Step>
            get() = allSteps.filter { it.shouldShow(showAll) }

        fun setSteps(steps: List<Step>, showAllSteps: Boolean = false) {
            allSteps = steps
            showAll = showAllSteps
            notifyDataSetChanged()
        }

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): StepViewHolder {
            val view = LayoutInflater.from(parent.context)
                .inflate(R.layout.item_step, parent, false)
            return StepViewHolder(view)
        }

        override fun onBindViewHolder(holder: StepViewHolder, position: Int) {
            holder.bind(filteredSteps[position])
        }

        override fun getItemCount(): Int = filteredSteps.size

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
                txtDescription.text = step.description

                // Color by importance
                val color = when (step.importance) {
                    "high" -> ContextCompat.getColor(itemView.context, android.R.color.holo_blue_dark)
                    "medium" -> ContextCompat.getColor(itemView.context, android.R.color.holo_green_dark)
                    "low" -> ContextCompat.getColor(itemView.context, android.R.color.darker_gray)
                    else -> ContextCompat.getColor(itemView.context, android.R.color.tertiary_text_light)
                }
                txtStepIndex.setTextColor(color)

                // Before/After
                if (step.before != null && step.after != null) {
                    txtBefore.text = step.before
                    txtAfter.text = step.after
                    txtBefore.visibility = View.VISIBLE
                    txtAfter.visibility = View.VISIBLE
                } else {
                    txtBefore.visibility = View.GONE
                    txtAfter.visibility = View.GONE
                }

                // Domain assumption
                if (step.domainAssumption != null) {
                    txtDomainAssumption.text = "⚠️ ${step.domainAssumption}"
                    txtDomainAssumption.visibility = View.VISIBLE
                } else {
                    txtDomainAssumption.visibility = View.GONE
                }
            }
        }
    }
}
