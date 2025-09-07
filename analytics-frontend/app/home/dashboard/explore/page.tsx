'use client'

import { useState, useEffect } from 'react'

interface ExpiryData {
  status: string
  message: string
  data: {
    total_licenses: number
    total_expired: number
    expired_percentage: number
    state_distribution: Record<string, number>
    specialty_distribution: Record<string, number>
    timeline_breakdown: {
      recently_expired: number
      expired_1_6_months: number
      expired_6_12_months: number
      expired_over_year: number
    }
    expired_details: Array<{
      license_number: string
      provider_name: string
      first_name_x: string
      last_name_x: string
      primary_specialty: string
      expiration_date: string
      days_expired: number
      address_state: string
    }>
  }
  processing_info: {
    merged_dataset_count: number
    cleaned_roster_count: number
  }
}

export default function ExplorePage() {
  const [expiryData, setExpiryData] = useState<ExpiryData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch expired license data from backend
  useEffect(() => {
    const fetchExpiryData = async () => {
      try {
        setLoading(true)
        const response = await fetch('http://localhost:8001/expiry')
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }
        
        const data: ExpiryData = await response.json()
        setExpiryData(data)
        setError(null)
      } catch (err) {
        console.error('Failed to fetch expiry data:', err)
        setError(err instanceof Error ? err.message : 'Failed to fetch data')
      } finally {
        setLoading(false)
      }
    }

    fetchExpiryData()
  }, [])

  // Loading state
  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <h3 className="text-xl font-semibold text-gray-800 mb-2">Loading Expired License Data</h3>
          <p className="text-gray-600">Fetching analytics from backend...</p>
        </div>
      </div>
    )
  }

  // Error state
  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="text-red-500 text-6xl mb-4">⚠️</div>
          <h3 className="text-xl font-semibold text-gray-800 mb-2">Failed to Load Data</h3>
          <p className="text-gray-600 mb-4">{error}</p>
          <button 
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  // No data state
  if (!expiryData) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="text-gray-400 text-6xl mb-4">📊</div>
          <h3 className="text-xl font-semibold text-gray-800 mb-2">No Data Available</h3>
          <p className="text-gray-600">No expired license data found.</p>
        </div>
      </div>
    )
  }

  // Helper functions to process API data
  const getStateDistributionData = () => {
    const stateData = Object.entries(expiryData.data.state_distribution)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 5)
    
    return stateData.map(([state, count]) => ({
      label: state,
      value: count
    }))
  }

  const getSpecialtyDistributionData = () => {
    const specialtyData = Object.entries(expiryData.data.specialty_distribution)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 5)
    
    return specialtyData.map(([specialty, count]) => ({
      label: specialty,
      value: count
    }))
  }

  const getTimelineData = () => {
    const timeline = expiryData.data.timeline_breakdown
    return [
      { label: "Recently Expired (≤30 days)", value: timeline.recently_expired },
      { label: "Expired 1-6 Months", value: timeline.expired_1_6_months },
      { label: "Expired 6-12 Months", value: timeline.expired_6_12_months },
      { label: "Expired Over 1 Year", value: timeline.expired_over_year }
    ]
  }
  
  // Pie chart component
  const PieChart = ({ title, data, colors }: { 
    title: string, 
    data: Array<{label: string, value: number}>, 
    colors: string[] 
  }) => {
    const total = data.reduce((sum, item) => sum + item.value, 0)
    let cumulativePercentage = 0
    
    return (
      <div className="bg-white rounded-2xl shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">{title}</h3>
        <div className="flex items-center justify-center">
          <div className="relative w-80 h-80">
            <svg className="w-80 h-80 transform -rotate-90" viewBox="0 0 100 100">
              {data.map((item, index) => {
                const percentage = (item.value / total) * 100
                const strokeDasharray = `${percentage} ${100 - percentage}`
                const strokeDashoffset = -cumulativePercentage
                cumulativePercentage += percentage
                
                return (
                  <circle
                    key={index}
                    cx="50"
                    cy="50"
                    r="30"
                    fill="transparent"
                    stroke={colors[index]}
                    strokeWidth="8"
                    strokeDasharray={strokeDasharray}
                    strokeDashoffset={strokeDashoffset}
                  />
                )
              })}
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-800">{total.toLocaleString()}</div>
                <div className="text-sm text-gray-500">Total</div>
              </div>
            </div>
          </div>
        </div>
        <div className="mt-4 space-y-2">
          {data.map((item, index) => (
            <div key={index} className="flex items-center justify-between">
              <div className="flex items-center">
                <div 
                  className="w-3 h-3 rounded-full mr-2" 
                  style={{ backgroundColor: colors[index] }}
                ></div>
                <span className="text-sm text-gray-600">{item.label}</span>
              </div>
              <span className="text-sm font-medium text-gray-800">
                {item.value.toLocaleString()} ({((item.value / total) * 100).toFixed(1)}%)
              </span>
            </div>
          ))}
        </div>
      </div>
    )
  }

  // Bar chart component
  const BarChart = ({ title, data }: { 
    title: string, 
    data: Array<{label: string, value: number}> 
  }) => {
    const maxValue = Math.max(...data.map(item => item.value))
    
    return (
      <div className="bg-white rounded-2xl shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-6">{title}</h3>
        <div className="space-y-4">
          {data.map((item, index) => (
            <div key={index} className="flex items-center">
              <div className="w-24 text-sm text-gray-600 text-right mr-4">
                {item.label}
              </div>
              <div className="flex-1 bg-gray-200 rounded-full h-6 relative">
                <div 
                  className="bg-gradient-to-r from-blue-500 to-blue-600 h-6 rounded-full flex items-center justify-end pr-2"
                  style={{ width: `${(item.value / maxValue) * 100}%` }}
                >
                  <span className="text-white text-xs font-medium">
                    {item.value.toLocaleString()}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        
        {/* Top Row - Two Number Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          {/* Total Expired License Card */}
          <div className="bg-white rounded-2xl shadow-lg p-8 text-center">
            <div className="text-4xl font-bold text-blue-600 mb-2">
              {expiryData.data.total_expired.toLocaleString()}
            </div>
            <div className="text-lg text-gray-600">Total Expired License</div>
            <div className="text-sm text-gray-500 mt-2">
              Providers with expired licenses
            </div>
          </div>

          {/* Expired License Percentage Card */}
          <div className="bg-white rounded-2xl shadow-lg p-8 text-center">
            <div className="text-4xl font-bold text-red-600 mb-2">
              {expiryData.data.expired_percentage.toFixed(1)}%
            </div>
            <div className="text-lg text-gray-600">Expired License Percentage</div>
            <div className="text-sm text-gray-500 mt-2">
              Percentage of providers with expired licenses
            </div>
          </div>
        </div>

        {/* Middle Row - Two Pie Chart Cards */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* State Wise Licenses Expiry Pie Chart */}
          <PieChart
            title="State Wise License Expiry"
            data={getStateDistributionData()}
            colors={["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]}
          />

          {/* Speciality Wise License Expiry Pie Chart */}
          <PieChart
            title="Speciality Wise License Expiry"
            data={getSpecialtyDistributionData()}
            colors={["#10B981", "#6B7280", "#F59E0B", "#EF4444", "#8B5CF6"]}
          />
        </div>

        {/* Bottom Row - Bar Chart Card */}
        <div className="grid grid-cols-1">
          <BarChart
            title="Timeline Breakdown"
            data={getTimelineData()}
          />
        </div>
      </div>
    </div>
  )
}
